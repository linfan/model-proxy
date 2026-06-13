use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use crate::record::Recorder;
use crate::types::anthropic::*;
use crate::types::config::ResolvedModel;
use crate::types::openai;

use super::convert;
use super::{proxy_error, ThinkingSettings};

// ── Direct Anthropic Messages API forwarding ──

pub async fn forward_messages(
    client: &Client,
    model: &ResolvedModel,
    mut body: Value,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
) -> Response {
    body["model"] = Value::String(model.entry.model.clone());

    if let Some(sp) = &model.entry.system_prompt {
        if !sp.is_empty() && body.get("system").is_none() {
            body["system"] = Value::String(sp.clone());
        }
    }

    if thinking.enabled && body.get("thinking").is_none() {
        let mut thinking_obj = serde_json::Map::new();
        thinking_obj.insert("type".to_string(), Value::String("enabled".to_string()));
        if let Some(budget) = thinking.budget_tokens {
            thinking_obj.insert("budget_tokens".to_string(), Value::Number(budget.into()));
        }
        body["thinking"] = Value::Object(thinking_obj);
    }

    debug!("Anthropic request: {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    recorder.record_request(request_id, &body).await;

    let resp = match client
        .post(&model.entry.url)
        .header("x-api-key", &model.entry.api_token)
        .header("anthropic-version", "2023-06-01")
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
        debug!("Anthropic error response: {err_body}");
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            err_body,
        )
            .into_response();
    }

    if is_stream {
        stream_anthropic_passthrough(resp, recorder.clone(), request_id.to_string()).await
    } else {
        non_stream_anthropic(resp, recorder, request_id).await
    }
}

// ── OpenAI chat completions → Anthropic backend ──

pub async fn forward_chat_via_anthropic(
    client: &Client,
    model: &ResolvedModel,
    request: &openai::ChatCompletionRequest,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
) -> Response {
    let anthropic_req = convert::openai_to_anthropic_request(request, &model.entry.model, thinking);
    let is_stream = request.stream.unwrap_or(false);

    if let Some(sp) = &model.entry.system_prompt {
        if !sp.is_empty() {
            // already handled in conversion if messages had system role
        }
    }

    let body = match serde_json::to_value(&anthropic_req) {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("serialize error: {e}")),
    };

    debug!("Anthropic request (via OpenAI endpoint): {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    recorder.record_request(request_id, &body).await;

    let resp = match client
        .post(&model.entry.url)
        .header("x-api-key", &model.entry.api_token)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("upstream error: {e}")),
    };

    let status = resp.status();

    if !status.is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        debug!("Anthropic error: {err_body}");
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            err_body,
        )
            .into_response();
    }

    if is_stream {
        stream_anthropic_to_openai(resp, recorder.clone(), request_id.to_string()).await
    } else {
        non_stream_anthropic_to_openai(resp, recorder, request_id).await
    }
}

// ── Anthropic Messages endpoint → OpenAI backend ──

pub async fn forward_messages_via_openai(
    client: &Client,
    model: &ResolvedModel,
    request: &MessagesRequest,
    recorder: &Recorder,
    request_id: &str,
) -> Response {
    let openai_req = convert::anthropic_to_openai_request(request, &model.entry.model);
    let is_stream = request.stream.unwrap_or(false);

    let body = match serde_json::to_value(&openai_req) {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("serialize error: {e}")),
    };

    debug!("OpenAI request (via Messages endpoint): {}", serde_json::to_string_pretty(&body).unwrap_or_default());

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

    if !status.is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            err_body,
        )
            .into_response();
    }

    if is_stream {
        stream_openai_to_anthropic(resp, recorder.clone(), request_id.to_string()).await
    } else {
        non_stream_openai_to_anthropic(resp, recorder, request_id).await
    }
}

// ── Non-stream handlers ──

async fn non_stream_anthropic(resp: reqwest::Response, recorder: &Recorder, request_id: &str) -> Response {
    let body: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}")),
    };

    debug!("Anthropic response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
    recorder.record_response(request_id, &body).await;

    axum::Json(body).into_response()
}

async fn non_stream_anthropic_to_openai(resp: reqwest::Response, recorder: &Recorder, request_id: &str) -> Response {
    let body: MessagesResponse = match resp.json().await {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}")),
    };

    let openai_resp = convert::anthropic_to_openai_response(&body);

    let resp_value = serde_json::to_value(&openai_resp).unwrap_or_default();
    recorder.record_response(request_id, &resp_value).await;

    axum::Json(openai_resp).into_response()
}

async fn non_stream_openai_to_anthropic(resp: reqwest::Response, recorder: &Recorder, request_id: &str) -> Response {
    let body: openai::ChatCompletionResponse = match resp.json().await {
        Ok(v) => v,
        Err(e) => return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}")),
    };

    let anthropic_resp = convert::openai_to_anthropic_response(&body);

    let resp_value = serde_json::to_value(&anthropic_resp).unwrap_or_default();
    recorder.record_response(request_id, &resp_value).await;

    axum::Json(anthropic_resp).into_response()
}

// ── Streaming: Anthropic SSE passthrough ──

async fn stream_anthropic_passthrough(resp: reqwest::Response, recorder: Recorder, request_id: String) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut last_data = Value::Null;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.starts_with("data:") {
                    let data = line.strip_prefix("data:").unwrap().trim();
                    if let Ok(parsed) = serde_json::from_str::<Value>(data) {
                        last_data = parsed;
                    }
                }

                let _ = tx.send(Ok(format!("{line}\n"))).await;
            }
        }

        if last_data != Value::Null {
            recorder.record_response(&request_id, &last_data).await;
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

// ── Streaming: Anthropic SSE → OpenAI SSE ──

async fn stream_anthropic_to_openai(resp: reqwest::Response, _recorder: Recorder, _request_id: String) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut pending_event_type: Option<String> = None;

        let mut msg_id = String::new();
        let mut msg_model = String::new();
        let mut _current_block_type: Option<String> = None;
        let mut response_text = String::new();
        let mut reasoning_text = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if let Some(event_name) = line.strip_prefix("event:") {
                    pending_event_type = Some(event_name.trim().to_string());
                    continue;
                }

                if !line.starts_with("data:") {
                    continue;
                }

                let data_str = line.strip_prefix("data:").unwrap().trim();
                let data: Value = match serde_json::from_str(data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = pending_event_type.take().unwrap_or_default();

                match event_type.as_str() {
                    "message_start" => {
                        if let Some(msg) = data.get("message") {
                            msg_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            msg_model = msg.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        }
                        let chunk = make_openai_chunk(&msg_id, &msg_model, Some("assistant"), None, None, None);
                        send_openai_chunk(&tx, &chunk).await;
                    }
                    "content_block_start" => {
                        if let Some(cb) = data.get("content_block") {
                            let cb_type = cb.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            _current_block_type = Some(cb_type.to_string());
                        }
                    }
                    "content_block_delta" => {
                        if let Some(delta) = data.get("delta") {
                            let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            match delta_type {
                                "text_delta" => {
                                    let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                    response_text.push_str(text);
                                    let chunk = make_openai_chunk(&msg_id, &msg_model, None, Some(text), None, None);
                                    send_openai_chunk(&tx, &chunk).await;
                                }
                                "thinking_delta" => {
                                    let thinking = delta.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                    reasoning_text.push_str(thinking);
                                    let chunk = make_openai_chunk(&msg_id, &msg_model, None, None, Some(thinking), None);
                                    send_openai_chunk(&tx, &chunk).await;
                                }
                                "input_json_delta" => {
                                    // Tool call argument streaming — simplified passthrough
                                }
                                _ => {}
                            }
                        }
                    }
                    "content_block_stop" => {
                        _current_block_type = None;
                    }
                    "message_delta" => {
                        let stop_reason = data
                            .get("delta")
                            .and_then(|d| d.get("stop_reason"))
                            .and_then(|v| v.as_str())
                            .map(|r| match r {
                                "end_turn" => "stop",
                                "max_tokens" => "length",
                                "tool_use" => "tool_calls",
                                other => other,
                            });

                        if let Some(reason) = stop_reason {
                            let chunk = make_openai_chunk_with_finish(&msg_id, &msg_model, reason);
                            send_openai_chunk(&tx, &chunk).await;
                        }

                        if let Some(usage) = data.get("usage") {
                            let output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let input_tokens = 0u32;
                            let chunk = make_openai_chunk_with_usage(&msg_id, &msg_model, input_tokens, output_tokens);
                            send_openai_chunk(&tx, &chunk).await;
                        }
                    }
                    "message_stop" => {
                        let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
                    }
                    _ => {}
                }
            }
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

// ── Streaming: OpenAI SSE → Anthropic SSE ──

async fn stream_openai_to_anthropic(resp: reqwest::Response, _recorder: Recorder, _request_id: String) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut first_chunk = true;
        let block_index: u32 = 0;
        let mut msg_id;
        let mut msg_model;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if !line.starts_with("data:") {
                    continue;
                }

                let data_str = line.strip_prefix("data:").unwrap().trim();

                if data_str == "[DONE]" {
                    send_anthropic_event(&tx, "content_block_stop", &serde_json::json!({"index": block_index})).await;
                    send_anthropic_event(&tx, "message_delta", &serde_json::json!({
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 0}
                    })).await;
                    send_anthropic_event(&tx, "message_stop", &serde_json::json!({})).await;
                    continue;
                }

                let data: Value = match serde_json::from_str(data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if first_chunk {
                    msg_id = data.get("id").and_then(|v| v.as_str()).unwrap_or("msg_proxy").to_string();
                    msg_model = data.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

                    send_anthropic_event(&tx, "message_start", &serde_json::json!({
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": msg_model,
                            "stop_reason": null,
                            "usage": {"input_tokens": 0, "output_tokens": 0}
                        }
                    })).await;

                    send_anthropic_event(&tx, "content_block_start", &serde_json::json!({
                        "index": 0,
                        "content_block": {"type": "text", "text": ""}
                    })).await;

                    first_chunk = false;
                }

                if let Some(choices) = data.get("choices").and_then(|c| c.as_array()) {
                    for choice in choices {
                        if let Some(delta) = choice.get("delta") {
                            if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                send_anthropic_event(&tx, "content_block_delta", &serde_json::json!({
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": content}
                                })).await;
                            }
                            if let Some(rc) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                                send_anthropic_event(&tx, "content_block_delta", &serde_json::json!({
                                    "index": block_index,
                                    "delta": {"type": "thinking_delta", "thinking": rc}
                                })).await;
                            }
                        }

                        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                            let stop = match reason {
                                "stop" => "end_turn",
                                "length" => "max_tokens",
                                "tool_calls" => "tool_use",
                                other => other,
                            };
                            send_anthropic_event(&tx, "content_block_stop", &serde_json::json!({"index": block_index})).await;
                            send_anthropic_event(&tx, "message_delta", &serde_json::json!({
                                "delta": {"stop_reason": stop},
                                "usage": {"output_tokens": 0}
                            })).await;
                            send_anthropic_event(&tx, "message_stop", &serde_json::json!({})).await;
                        }
                    }
                }
            }
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

// ── Helpers ──

fn make_openai_chunk(id: &str, model: &str, role: Option<&str>, content: Option<&str>, reasoning: Option<&str>, finish: Option<&str>) -> Value {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut delta = serde_json::Map::new();
    if let Some(r) = role {
        delta.insert("role".to_string(), Value::String(r.to_string()));
    }
    if let Some(c) = content {
        delta.insert("content".to_string(), Value::String(c.to_string()));
    }
    if let Some(r) = reasoning {
        delta.insert("reasoning_content".to_string(), Value::String(r.to_string()));
    }

    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish
        }]
    })
}

fn make_openai_chunk_with_finish(id: &str, model: &str, reason: &str) -> Value {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": reason
        }]
    })
}

fn make_openai_chunk_with_usage(id: &str, model: &str, input: u32, output: u32) -> Value {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": input,
            "completion_tokens": output,
            "total_tokens": input + output
        }
    })
}

async fn send_openai_chunk(tx: &tokio::sync::mpsc::Sender<Result<String, std::io::Error>>, chunk: &Value) {
    let data = serde_json::to_string(chunk).unwrap_or_default();
    let _ = tx.send(Ok(format!("data: {data}\n\n"))).await;
}

async fn send_anthropic_event(tx: &tokio::sync::mpsc::Sender<Result<String, std::io::Error>>, event: &str, data: &Value) {
    let data_str = serde_json::to_string(data).unwrap_or_default();
    let _ = tx.send(Ok(format!("event: {event}\ndata: {data_str}\n\n"))).await;
}
