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
    session_id: Option<&str>,
) -> Response {
    body["model"] = Value::String(model.entry.model.clone());

    if let Some(sp) = &model.entry.system_prompt {
        if !sp.is_empty() && body.get("system").is_none() {
            body["system"] = Value::String(sp.clone());
        }
    }

    if thinking.enabled == Some(true) && body.get("thinking").is_none() {
        let mut thinking_obj = serde_json::Map::new();
        thinking_obj.insert("type".to_string(), Value::String("enabled".to_string()));
        if let Some(budget) = thinking.budget_tokens {
            thinking_obj.insert("budget_tokens".to_string(), Value::Number(budget.into()));
        }
        body["thinking"] = Value::Object(thinking_obj);
    }

    debug!("Anthropic request: {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    let req_headers = format!(
        "POST {}\nx-api-key: {}\nanthropic-version: 2023-06-01\nContent-Type: application/json",
        &model.entry.url, &model.entry.api_token
    );
    recorder.record_request(request_id, session_id, &req_headers, &body).await;

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
        Err(e) => {
            let msg = format!("upstream error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &msg);
        }
    };

    let status = resp.status();
    let is_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    if !status.is_success() {
        let err_headers = format_response_headers(&resp);
        let err_body = resp.text().await.unwrap_or_default();
        recorder.record_error_response(request_id, session_id, &err_headers, &err_body).await;
        debug!("Anthropic error response: {err_body}");
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
        stream_anthropic_passthrough(resp, recorder.clone(), request_id.to_string(), resp_headers, sid).await
    } else {
        non_stream_anthropic(resp, recorder, request_id, resp_headers, sid.as_deref()).await
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
    session_id: Option<&str>,
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
        Err(e) => {
            let msg = format!("serialize error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &msg);
        }
    };

    debug!("Anthropic request (via OpenAI endpoint): {}", serde_json::to_string_pretty(&body).unwrap_or_default());

    let req_headers = format!(
        "POST {}\nx-api-key: {}\nanthropic-version: 2023-06-01\nContent-Type: application/json",
        &model.entry.url, &model.entry.api_token
    );
    recorder.record_request(request_id, session_id, &req_headers, &body).await;

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
        Err(e) => {
            let msg = format!("upstream error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &msg);
        }
    };

    let status = resp.status();

    if !status.is_success() {
        let err_headers = format_response_headers(&resp);
        let err_body = resp.text().await.unwrap_or_default();
        recorder.record_error_response(request_id, session_id, &err_headers, &err_body).await;
        debug!("Anthropic error: {err_body}");
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
        stream_anthropic_to_openai(resp, recorder.clone(), request_id.to_string(), resp_headers, sid).await
    } else {
        non_stream_anthropic_to_openai(resp, recorder, request_id, resp_headers, sid.as_deref()).await
    }
}

// ── Anthropic Messages endpoint → OpenAI backend ──

pub async fn forward_messages_via_openai(
    client: &Client,
    model: &ResolvedModel,
    request: &MessagesRequest,
    recorder: &Recorder,
    request_id: &str,
    session_id: Option<&str>,
) -> Response {
    let openai_req = convert::anthropic_to_openai_request(request, &model.entry.model);
    let is_stream = request.stream.unwrap_or(false);

    let body = match serde_json::to_value(&openai_req) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("serialize error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &msg);
        }
    };

    debug!("OpenAI request (via Messages endpoint): {}", serde_json::to_string_pretty(&body).unwrap_or_default());

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
        Err(e) => {
            let msg = format!("upstream error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &msg);
        }
    };

    let status = resp.status();

    if !status.is_success() {
        let err_headers = format_response_headers(&resp);
        let err_body = resp.text().await.unwrap_or_default();
        recorder.record_error_response(request_id, session_id, &err_headers, &err_body).await;
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
        stream_openai_to_anthropic(resp, recorder.clone(), request_id.to_string(), resp_headers, sid).await
    } else {
        non_stream_openai_to_anthropic(resp, recorder, request_id, resp_headers, sid.as_deref()).await
    }
}

// ── Non-stream handlers ──

async fn non_stream_anthropic(resp: reqwest::Response, recorder: &Recorder, request_id: &str, resp_headers: String, session_id: Option<&str>) -> Response {
    let raw = read_body(resp, recorder, request_id, session_id).await;
    let raw = match raw {
        Ok(raw) => raw,
        Err(resp) => return resp,
    };
    let body: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            recorder.record_error_response(request_id, session_id, &resp_headers, &raw).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}"));
        }
    };

    debug!("Anthropic response: {}", serde_json::to_string_pretty(&body).unwrap_or_default());
    recorder.record_response(request_id, session_id, &resp_headers, &body).await;

    axum::Json(body).into_response()
}

async fn non_stream_anthropic_to_openai(resp: reqwest::Response, recorder: &Recorder, request_id: &str, resp_headers: String, session_id: Option<&str>) -> Response {
    let raw = read_body(resp, recorder, request_id, session_id).await;
    let raw = match raw {
        Ok(raw) => raw,
        Err(resp) => return resp,
    };
    let body: MessagesResponse = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            recorder.record_error_response(request_id, session_id, &resp_headers, &raw).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}"));
        }
    };

    let openai_resp = convert::anthropic_to_openai_response(&body);

    let resp_value = serde_json::to_value(&openai_resp).unwrap_or_default();
    recorder.record_response(request_id, session_id, &resp_headers, &resp_value).await;

    axum::Json(openai_resp).into_response()
}

async fn non_stream_openai_to_anthropic(resp: reqwest::Response, recorder: &Recorder, request_id: &str, resp_headers: String, session_id: Option<&str>) -> Response {
    let raw = read_body(resp, recorder, request_id, session_id).await;
    let raw = match raw {
        Ok(raw) => raw,
        Err(resp) => return resp,
    };
    let body: openai::ChatCompletionResponse = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            recorder.record_error_response(request_id, session_id, &resp_headers, &raw).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}"));
        }
    };

    let anthropic_resp = convert::openai_to_anthropic_response(&body);

    let resp_value = serde_json::to_value(&anthropic_resp).unwrap_or_default();
    recorder.record_response(request_id, session_id, &resp_headers, &resp_value).await;

    axum::Json(anthropic_resp).into_response()
}

/// Read the upstream body, recording a no-response error if reading fails.
async fn read_body(resp: reqwest::Response, recorder: &Recorder, request_id: &str, session_id: Option<&str>) -> Result<String, Response> {
    match resp.text().await {
        Ok(raw) => Ok(raw),
        Err(e) => {
            let msg = format!("read response body error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            Err(proxy_error(StatusCode::BAD_GATEWAY, &msg))
        }
    }
}

// ── Streaming: Anthropic SSE passthrough ──

async fn stream_anthropic_passthrough(resp: reqwest::Response, recorder: Recorder, request_id: String, resp_headers: String, session_id: Option<String>) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut pending_event_type: Option<String> = None;

        let mut msg_id = String::new();
        let mut msg_model = String::new();
        let mut msg_role = String::new();
        let mut content_blocks: Vec<Value> = Vec::new();
        let mut current_block: Option<Value> = None;
        let mut stop_reason: Option<String> = None;
        let mut usage = serde_json::json!({"input_tokens": 0, "output_tokens": 0});
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

                if let Some(event_name) = line.strip_prefix("event:") {
                    pending_event_type = Some(event_name.trim().to_string());
                }

                if line.starts_with("data:") {
                    let data_str = line.strip_prefix("data:").unwrap().trim();
                    if let Ok(parsed) = serde_json::from_str::<Value>(data_str) {
                        has_data = true;
                        let event_type = pending_event_type.take().unwrap_or_default();

                        match event_type.as_str() {
                            "message_start" => {
                                if let Some(msg) = parsed.get("message") {
                                    msg_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    msg_model = msg.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    msg_role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("assistant").to_string();
                                    if let Some(u) = msg.get("usage") {
                                        usage = u.clone();
                                    }
                                }
                            }
                            "content_block_start" => {
                                if let Some(cb) = parsed.get("content_block") {
                                    current_block = Some(cb.clone());
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = parsed.get("delta") {
                                    let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                    if let Some(ref mut block) = current_block {
                                        match delta_type {
                                            "text_delta" => {
                                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                                    let cur = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                                    block["text"] = Value::String(format!("{cur}{text}"));
                                                }
                                            }
                                            "thinking_delta" => {
                                                if let Some(thinking) = delta.get("thinking").and_then(|v| v.as_str()) {
                                                    let cur = block.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
                                                    block["thinking"] = Value::String(format!("{cur}{thinking}"));
                                                }
                                            }
                                            "input_json_delta" => {
                                                if let Some(json_str) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                                    let cur = block.get("_partial_input").and_then(|v| v.as_str()).unwrap_or("");
                                                    block["_partial_input"] = Value::String(format!("{cur}{json_str}"));
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            "content_block_stop" => {
                                if let Some(mut block) = current_block.take() {
                                    if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
                                        if let Some(partial) = block.get("_partial_input").and_then(|v| v.as_str()).map(|s| s.to_string()) {
                                            if let Ok(input) = serde_json::from_str::<Value>(&partial) {
                                                block["input"] = input;
                                            }
                                            block.as_object_mut().map(|m| m.remove("_partial_input"));
                                        }
                                    }
                                    content_blocks.push(block);
                                }
                            }
                            "message_delta" => {
                                if let Some(delta) = parsed.get("delta") {
                                    if let Some(sr) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                                        stop_reason = Some(sr.to_string());
                                    }
                                }
                                if let Some(u) = parsed.get("usage") {
                                    if let Some(ot) = u.get("output_tokens") {
                                        usage["output_tokens"] = ot.clone();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }

                let _ = tx.send(Ok(format!("{line}\n"))).await;
            }
        }

        if has_data {
            let assembled = serde_json::json!({
                "id": msg_id,
                "type": "message",
                "role": msg_role,
                "model": msg_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": usage
            });
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

// ── Streaming: Anthropic SSE → OpenAI SSE ──

async fn stream_anthropic_to_openai(resp: reqwest::Response, recorder: Recorder, request_id: String, resp_headers: String, session_id: Option<String>) -> Response {
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
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut current_tool_call: Option<Value> = None;
        let mut current_tool_args = String::new();
        let mut finish_reason: Option<String> = None;
        let mut usage_value: Option<Value> = None;
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
                has_data = true;

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
                            if cb_type == "tool_use" {
                                current_tool_call = Some(serde_json::json!({
                                    "id": cb.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                    "type": "function",
                                    "function": {
                                        "name": cb.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                        "arguments": ""
                                    }
                                }));
                                current_tool_args.clear();
                            }
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
                                    if let Some(json_str) = delta.get("partial_json").and_then(|v| v.as_str()) {
                                        current_tool_args.push_str(json_str);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    "content_block_stop" => {
                        if let Some(mut tc) = current_tool_call.take() {
                            tc["function"]["arguments"] = Value::String(current_tool_args.clone());
                            tool_calls.push(tc);
                            current_tool_args.clear();
                        }
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
                            finish_reason = Some(reason.to_string());
                            let chunk = make_openai_chunk_with_finish(&msg_id, &msg_model, reason);
                            send_openai_chunk(&tx, &chunk).await;
                        }

                        if let Some(usage) = data.get("usage") {
                            let output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                            let input_tokens = 0u32;
                            usage_value = Some(serde_json::json!({
                                "prompt_tokens": input_tokens,
                                "completion_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens
                            }));
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

        if has_data {
            let assembled = assemble_openai_response(
                &msg_id, &msg_model,
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

// ── Streaming: OpenAI SSE → Anthropic SSE ──

async fn stream_openai_to_anthropic(resp: reqwest::Response, recorder: Recorder, request_id: String, resp_headers: String, session_id: Option<String>) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();
        let mut first_chunk = true;
        let block_index: u32 = 0;
        let mut msg_id = String::new();
        let mut msg_model = String::new();
        let mut response_text = String::new();
        let mut reasoning_text = String::new();
        let mut stop_reason: Option<String> = None;
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

                has_data = true;

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
                                response_text.push_str(content);
                                send_anthropic_event(&tx, "content_block_delta", &serde_json::json!({
                                    "index": block_index,
                                    "delta": {"type": "text_delta", "text": content}
                                })).await;
                            }
                            if let Some(rc) = delta.get("reasoning_content").and_then(|c| c.as_str()) {
                                reasoning_text.push_str(rc);
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
                            stop_reason = Some(stop.to_string());
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

        if has_data {
            let mut content_blocks: Vec<Value> = Vec::new();
            if !reasoning_text.is_empty() {
                content_blocks.push(serde_json::json!({"type": "thinking", "thinking": reasoning_text}));
            }
            if !response_text.is_empty() {
                content_blocks.push(serde_json::json!({"type": "text", "text": response_text}));
            }
            let assembled = serde_json::json!({
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": msg_model,
                "content": content_blocks,
                "stop_reason": stop_reason,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            });
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

// ── Helpers ──

fn assemble_openai_response(
    id: &str, model: &str,
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

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut resp = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": now,
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

fn format_response_headers(resp: &reqwest::Response) -> String {
    let mut lines = vec![format!("HTTP/1.1 {}", resp.status())];
    for (name, value) in resp.headers() {
        if let Ok(v) = value.to_str() {
            lines.push(format!("{}: {}", name, v));
        }
    }
    lines.join("\n")
}
