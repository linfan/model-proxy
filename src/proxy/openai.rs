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
use crate::types::openai::*;

use super::{proxy_error, ThinkingSettings};

pub async fn forward_chat_completion(
    client: &Client,
    model: &ResolvedModel,
    mut request: ChatCompletionRequest,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
) -> Response {
    request.model = model.entry.model.clone();

    if let Some(sp) = &model.entry.system_prompt {
        if !sp.is_empty() {
            inject_system_prompt(&mut request, sp);
        }
    }

    // Remove the thinking override from the request body — it's not an OpenAI parameter
    request.thinking = None;

    let body = match serde_json::to_value(&request) {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("serialize error: {e}")),
    };

    debug!("OpenAI request: {} {}\n{}", "POST", &model.entry.url, serde_json::to_string_pretty(&body).unwrap_or_default());

    recorder.record_request(request_id, &body).await;

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
    let is_stream = request.stream.unwrap_or(false);

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

    if is_stream {
        stream_response(resp, recorder.clone(), request_id.to_string(), thinking.enabled).await
    } else {
        non_stream_response(resp, recorder, request_id).await
    }
}

async fn non_stream_response(resp: reqwest::Response, recorder: &Recorder, request_id: &str) -> Response {
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}")),
    };

    debug!("OpenAI response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
    recorder.record_response(request_id, &body).await;

    axum::Json(body).into_response()
}

async fn stream_response(resp: reqwest::Response, recorder: Recorder, request_id: String, parse_think_tags: bool) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut collected_response = Value::Null;
        let mut response_text = String::new();
        let mut reasoning_text = String::new();
        let mut in_think_block = false;

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
                        if parse_think_tags {
                            process_think_tags(&mut event, &mut in_think_block, &mut response_text, &mut reasoning_text);
                        }
                        collected_response = event.clone();

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

        if collected_response != Value::Null {
            recorder.record_response(&request_id, &collected_response).await;
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

fn inject_system_prompt(request: &mut ChatCompletionRequest, prompt: &str) {
    if request.messages.is_empty() {
        return;
    }
    if request.messages[0].role == "user" {
        request.messages.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: Some(Value::String(prompt.to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
                extra: Default::default(),
            },
        );
    }
}
