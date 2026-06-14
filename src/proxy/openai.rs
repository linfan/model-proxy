use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use crate::record::Recorder;
use crate::types::config::ResolvedModel;

use super::{proxy_error, ThinkingSettings};

pub async fn forward_chat_completion(
    client: &Client,
    model: &ResolvedModel,
    mut body: Value,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
    session_id: Option<&str>,
) -> Response {
    body["model"] = Value::String(model.entry.model.clone());

    if let Some(sp) = &model.entry.system_prompt {
        if !sp.is_empty() {
            inject_system_prompt(&mut body, sp);
        }
    }

    if let Some(enabled) = thinking.enabled {
        if body.get("enable_thinking").is_none() {
            body["enable_thinking"] = Value::Bool(enabled);
        }
        if enabled {
            if let Some(budget) = thinking.budget_tokens {
                if body.get("thinking_budget").is_none() {
                    body["thinking_budget"] = Value::Number(budget.into());
                }
            }
        }
    }

    debug!("OpenAI request: {} {}\n{}", "POST", &model.entry.url, serde_json::to_string_pretty(&body).unwrap_or_default());

    let req_headers = format!(
        "POST {}\nAuthorization: Bearer {}\nContent-Type: application/json",
        &model.entry.url, &model.entry.api_token
    );
    recorder.record_request(request_id, session_id, &req_headers, &body).await;

    let resp = match client
        .post(&model.entry.url)
        .header("Authorization", format!("Bearer {}", model.entry.api_token))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("upstream error: {e}")),
    };

    let status = resp.status();
    let is_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    if !status.is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        debug!("OpenAI error response: {err_body}");
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            err_body,
        )
            .into_response();
    }

    let resp_headers = format_response_headers(&resp);
    let sid = session_id.map(|s| s.to_string());
    if is_stream {
        stream_response(resp, recorder.clone(), request_id.to_string(), thinking.enabled == Some(true), resp_headers, sid).await
    } else {
        non_stream_response_with_headers(resp, recorder, request_id, resp_headers, sid.as_deref()).await
    }
}

async fn non_stream_response_with_headers(resp: reqwest::Response, recorder: &Recorder, request_id: &str, resp_headers: String, session_id: Option<&str>) -> Response {
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}")),
    };

    debug!("OpenAI response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
    recorder.record_response(request_id, session_id, &resp_headers, &body).await;

    axum::Json(body).into_response()
}

async fn stream_response(resp: reqwest::Response, recorder: Recorder, request_id: String, parse_think_tags: bool, resp_headers: String, session_id: Option<String>) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut response_text = String::new();
        let mut reasoning_text = String::new();
        let mut in_think_block = false;

        let mut chunk_id = String::new();
        let mut chunk_model = String::new();
        let mut chunk_created: u64 = 0;
        let mut finish_reason: Option<String> = None;
        let mut usage_value: Option<Value> = None;
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut has_data = false;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() {
                    let _ = tx.send(Ok("\n".to_string())).await;
                    continue;
                }

                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim();

                    if data == "[DONE]" {
                        let _ = tx.send(Ok("data: [DONE]\n".to_string())).await;
                        continue;
                    }

                    if let Ok(mut event) = serde_json::from_str::<Value>(data) {
                        if chunk_id.is_empty() {
                            chunk_id = event.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            chunk_model = event.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            chunk_created = event.get("created").and_then(|v| v.as_u64()).unwrap_or(0);
                        }
                        has_data = true;

                        if parse_think_tags {
                            process_think_tags(&mut event, &mut in_think_block, &mut response_text, &mut reasoning_text);
                        }

                        if let Some(choices) = event.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                if let Some(delta) = choice.get("delta") {
                                    if !parse_think_tags {
                                        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                                            response_text.push_str(c);
                                        }
                                        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                                            reasoning_text.push_str(r);
                                        }
                                    }
                                    if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                        accumulate_tool_calls(&mut tool_calls, tcs);
                                    }
                                }
                                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                                    finish_reason = Some(fr.to_string());
                                }
                            }
                        }

                        if let Some(u) = event.get("usage").filter(|u| u.is_object()) {
                            usage_value = Some(u.clone());
                        }

                        let modified_data = serde_json::to_string(&event).unwrap_or_default();
                        let _ = tx.send(Ok(format!("data: {modified_data}\n"))).await;
                    } else {
                        let _ = tx.send(Ok(format!("{line}\n"))).await;
                    }
                } else {
                    let _ = tx.send(Ok(format!("{line}\n"))).await;
                }
            }
        }

        if has_data {
            let assembled = assemble_openai_response(
                &chunk_id, &chunk_model, chunk_created,
                &response_text, &reasoning_text, &tool_calls,
                finish_reason.as_deref(), usage_value,
            );
            recorder.record_response(&request_id, session_id.as_deref(), &resp_headers, &assembled).await;
        }
    });

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn process_think_tags(
    event: &mut Value,
    in_think_block: &mut bool,
    response_text: &mut String,
    reasoning_text: &mut String,
) {
    let choices = match event.get_mut("choices").and_then(|c| c.as_array_mut()) {
        Some(c) => c,
        None => return,
    };

    for choice in choices {
        let delta = match choice.get_mut("delta") {
            Some(d) => d,
            None => continue,
        };

        if let Some(content) = delta.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()) {
            if content == "<think>" {
                delta.as_object_mut().map(|d| d.remove("content"));
                *in_think_block = true;
            } else if content == "</think>" {
                delta.as_object_mut().map(|d| d.remove("content"));
                *in_think_block = false;
            } else if *in_think_block {
                delta.as_object_mut().map(|d| {
                    d.remove("content");
                    d.insert("reasoning_content".to_string(), Value::String(content.clone()));
                });
                reasoning_text.push_str(&content);
            } else if content.contains("<think>") {
                handle_inline_think_tags(delta, &content, in_think_block, response_text, reasoning_text);
            } else if content.contains("</think>") {
                handle_inline_close_tags(delta, &content, in_think_block, response_text, reasoning_text);
            } else {
                response_text.push_str(&content);
            }
        }

        if let Some(rc) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
            reasoning_text.push_str(rc);
        }
    }
}

fn handle_inline_think_tags(
    delta: &mut Value,
    content: &str,
    in_think_block: &mut bool,
    response_text: &mut String,
    reasoning_text: &mut String,
) {
    let parts: Vec<&str> = content.splitn(2, "<think>").collect();
    let before = parts[0];
    let after = if parts.len() > 1 { parts[1] } else { "" };

    if !before.is_empty() {
        response_text.push_str(before);
    }

    *in_think_block = true;

    if after.contains("</think>") {
        let close_parts: Vec<&str> = after.splitn(2, "</think>").collect();
        let thinking = close_parts[0];
        let after_close = if close_parts.len() > 1 { close_parts[1] } else { "" };
        reasoning_text.push_str(thinking);
        *in_think_block = false;
        if !after_close.is_empty() {
            response_text.push_str(after_close);
        }
    } else {
        reasoning_text.push_str(after);
    }

    delta.as_object_mut().map(|d| d.remove("content"));
}

fn handle_inline_close_tags(
    delta: &mut Value,
    content: &str,
    in_think_block: &mut bool,
    response_text: &mut String,
    reasoning_text: &mut String,
) {
    let parts: Vec<&str> = content.splitn(2, "</think>").collect();
    let before = parts[0];
    let after = if parts.len() > 1 { parts[1] } else { "" };

    reasoning_text.push_str(before);
    *in_think_block = false;

    if !after.is_empty() {
        response_text.push_str(after);
    }

    delta.as_object_mut().map(|d| d.remove("content"));
}

fn accumulate_tool_calls(tool_calls: &mut Vec<Value>, delta_tcs: &[Value]) {
    for tc in delta_tcs {
        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        while tool_calls.len() <= idx {
            tool_calls.push(serde_json::json!({}));
        }

        let existing = &mut tool_calls[idx];

        if let Some(id) = tc.get("id") {
            existing["id"] = id.clone();
        }
        if let Some(t) = tc.get("type") {
            existing["type"] = t.clone();
        }
        if let Some(func) = tc.get("function") {
            if existing.get("function").is_none() {
                existing["function"] = serde_json::json!({});
            }
            if let Some(name) = func.get("name") {
                existing["function"]["name"] = name.clone();
            }
            if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                let current = existing["function"]["arguments"].as_str().unwrap_or("");
                existing["function"]["arguments"] = Value::String(format!("{current}{args}"));
            }
        }
    }
}

fn assemble_openai_response(
    id: &str, model: &str, created: u64,
    content: &str, reasoning: &str, tool_calls: &[Value],
    finish_reason: Option<&str>, usage: Option<Value>,
) -> Value {
    let mut message = serde_json::Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));
    if !content.is_empty() {
        message.insert("content".to_string(), Value::String(content.to_string()));
    }
    if !reasoning.is_empty() {
        message.insert("reasoning_content".to_string(), Value::String(reasoning.to_string()));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_string(), Value::Array(tool_calls.to_vec()));
    }

    let mut resp = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }]
    });

    if let Some(u) = usage {
        resp["usage"] = u;
    }

    resp
}

fn format_response_headers(resp: &reqwest::Response) -> String {
    let mut lines = vec![format!("HTTP/1.1 {}", resp.status())];
    for (name, value) in resp.headers() {
        if let Ok(v) = value.to_str() {
            lines.push(format!("{}: {}", name, v));
        }
    }
    lines.join("\n")
}

fn inject_system_prompt(body: &mut Value, prompt: &str) {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return,
    };
    if messages.is_empty() {
        return;
    }
    if messages[0].get("role").and_then(|r| r.as_str()) == Some("user") {
        messages.insert(0, serde_json::json!({
            "role": "system",
            "content": prompt,
        }));
    }
}
