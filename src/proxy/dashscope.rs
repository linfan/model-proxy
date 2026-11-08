use std::collections::HashMap;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Map, Value};
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use crate::record::Recorder;
use crate::types::anthropic::MessagesRequest;
use crate::types::config::ResolvedModel;
use crate::types::openai::ChatCompletionResponse;

use super::convert;
use super::{proxy_error, ThinkingSettings};

// ── Inbound: OpenAI chat completions → DashScope backend ──

pub async fn forward_chat_completion(
    client: &Client,
    model: &ResolvedModel,
    body: Value,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
    session_id: Option<&str>,
) -> Response {
    let is_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let ds_body = build_request(
        &body,
        &model.entry.model,
        thinking,
        model.entry.system_prompt.as_deref(),
        is_stream,
    );

    send_dashscope(
        client,
        model,
        ds_body,
        is_stream,
        OutputFormat::OpenAI,
        recorder,
        request_id,
        session_id,
    )
    .await
}

// ── Inbound: Anthropic messages → DashScope backend ──

pub async fn forward_messages(
    client: &Client,
    model: &ResolvedModel,
    request: &MessagesRequest,
    thinking: &ThinkingSettings,
    recorder: &Recorder,
    request_id: &str,
    session_id: Option<&str>,
) -> Response {
    let openai_req = convert::anthropic_to_openai_request(request, &model.entry.model);
    let body = match serde_json::to_value(&openai_req) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("serialize error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::INTERNAL_SERVER_ERROR, &msg);
        }
    };

    let effective_thinking = match &request.thinking {
        Some(t) => ThinkingSettings {
            enabled: Some(t.thinking_type == "enabled"),
            budget_tokens: t.budget_tokens,
        },
        None => thinking.clone(),
    };

    let is_stream = request.stream.unwrap_or(false);
    let ds_body = build_request(
        &body,
        &model.entry.model,
        &effective_thinking,
        model.entry.system_prompt.as_deref(),
        is_stream,
    );

    send_dashscope(
        client,
        model,
        ds_body,
        is_stream,
        OutputFormat::Anthropic,
        recorder,
        request_id,
        session_id,
    )
    .await
}

// ── Request conversion ──

fn build_request(
    body: &Value,
    remote_model: &str,
    thinking: &ThinkingSettings,
    system_prompt: Option<&str>,
    is_stream: bool,
) -> Value {
    let mut messages: Vec<Value> = body
        .get("messages")
        .and_then(|m| m.as_array())
        .map(|msgs| msgs.iter().map(convert_message).collect())
        .unwrap_or_default();

    if let Some(sp) = system_prompt {
        if !sp.is_empty()
            && messages
                .first()
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
                == Some("user")
        {
            messages.insert(0, json!({"role": "system", "content": sp}));
        }
    }

    let mut parameters = Map::new();
    parameters.insert("result_format".to_string(), json!("message"));
    if is_stream {
        parameters.insert("incremental_output".to_string(), json!(true));
    }

    for key in [
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "response_format",
        "tools",
        "tool_choice",
        "reasoning_effort",
    ] {
        if let Some(v) = body.get(key) {
            if !v.is_null() {
                parameters.insert(key.to_string(), v.clone());
            }
        }
    }

    if let Some(v) = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
    {
        parameters.insert("max_tokens".to_string(), v.clone());
    }

    let client_enable = body.get("enable_thinking").and_then(|v| v.as_bool());
    let enable = client_enable.or(thinking.enabled);
    if let Some(enabled) = enable {
        parameters.insert("enable_thinking".to_string(), json!(enabled));
        if enabled {
            let budget = body
                .get("thinking_budget")
                .and_then(|v| v.as_u64())
                .map(|b| b as u32)
                .or(thinking.budget_tokens);
            if let Some(b) = budget {
                parameters.insert("thinking_budget".to_string(), json!(b));
            }
        }
    }

    json!({
        "model": remote_model,
        "input": { "messages": messages },
        "parameters": parameters,
    })
}

fn convert_message(msg: &Value) -> Value {
    let mut out = Map::new();

    if let Some(role) = msg.get("role") {
        out.insert("role".to_string(), role.clone());
    }
    if let Some(content) = msg.get("content") {
        out.insert("content".to_string(), convert_content(content));
    }
    for key in ["tool_calls", "tool_call_id", "reasoning_content"] {
        if let Some(v) = msg.get(key) {
            if !v.is_null() {
                out.insert(key.to_string(), v.clone());
            }
        }
    }

    Value::Object(out)
}

fn convert_content(content: &Value) -> Value {
    let parts = match content.as_array() {
        Some(p) => p,
        None => return content.clone(),
    };

    let mut texts: Vec<String> = Vec::new();
    let mut media_parts: Vec<Value> = Vec::new();
    let mut has_media = false;

    for part in parts {
        match part.get("type").and_then(|t| t.as_str()) {
            Some("text") | Some("input_text") => {
                let text = part
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string();
                texts.push(text.clone());
                media_parts.push(json!({"text": text}));
            }
            Some("image_url") | Some("input_image") => {
                has_media = true;
                let url = part
                    .get("image_url")
                    .and_then(|i| i.get("url"))
                    .and_then(|u| u.as_str())
                    .or_else(|| part.get("image_url").and_then(|u| u.as_str()))
                    .or_else(|| part.get("image").and_then(|u| u.as_str()))
                    .unwrap_or_default();
                media_parts.push(json!({"image": url}));
            }
            _ => {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    texts.push(text.to_string());
                    media_parts.push(json!({"text": text}));
                }
            }
        }
    }

    if has_media {
        Value::Array(media_parts)
    } else {
        Value::String(texts.join(""))
    }
}

// ── Upstream call ──

#[derive(Clone, Copy, PartialEq)]
enum OutputFormat {
    OpenAI,
    Anthropic,
}

async fn send_dashscope(
    client: &Client,
    model: &ResolvedModel,
    body: Value,
    is_stream: bool,
    format: OutputFormat,
    recorder: &Recorder,
    request_id: &str,
    session_id: Option<&str>,
) -> Response {
    debug!(
        "DashScope request: POST {}\n{}",
        &model.entry.url,
        serde_json::to_string_pretty(&body).unwrap_or_default()
    );

    let sse_header = if is_stream {
        "\nX-DashScope-SSE: enable"
    } else {
        ""
    };
    let req_headers = format!(
        "POST {}\nAuthorization: Bearer {}{}\nContent-Type: application/json",
        &model.entry.url, &model.entry.api_token, sse_header
    );
    recorder
        .record_request(request_id, session_id, &req_headers, &body)
        .await;

    let mut request = client
        .post(&model.entry.url)
        .header("Authorization", format!("Bearer {}", model.entry.api_token))
        .header("Content-Type", "application/json");
    if is_stream {
        request = request.header("X-DashScope-SSE", "enable");
    }

    let resp = match request.json(&body).send().await {
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
        recorder
            .record_error_response(request_id, session_id, &err_headers, &err_body)
            .await;
        debug!("DashScope error response: {err_body}");
        return (
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            err_body,
        )
            .into_response();
    }

    let resp_headers = format_response_headers(&resp);
    let sid = session_id.map(|s| s.to_string());
    let remote_model = model.entry.model.clone();

    if is_stream {
        match format {
            OutputFormat::OpenAI => {
                stream_to_openai(
                    resp,
                    recorder.clone(),
                    request_id.to_string(),
                    resp_headers,
                    sid,
                    remote_model,
                )
                .await
            }
            OutputFormat::Anthropic => {
                stream_to_anthropic(
                    resp,
                    recorder.clone(),
                    request_id.to_string(),
                    resp_headers,
                    sid,
                    remote_model,
                )
                .await
            }
        }
    } else {
        non_stream(
            resp,
            format,
            recorder,
            request_id,
            resp_headers,
            sid.as_deref(),
            &remote_model,
        )
        .await
    }
}

// ── Non-streaming response ──

async fn non_stream(
    resp: reqwest::Response,
    format: OutputFormat,
    recorder: &Recorder,
    request_id: &str,
    resp_headers: String,
    session_id: Option<&str>,
    remote_model: &str,
) -> Response {
    let raw = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("read response body error: {e}");
            recorder.record_error_reason(request_id, session_id, &msg).await;
            return proxy_error(StatusCode::BAD_GATEWAY, &msg);
        }
    };

    let ds_body: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            recorder
                .record_error_response(request_id, session_id, &resp_headers, &raw)
                .await;
            return proxy_error(StatusCode::BAD_GATEWAY, &format!("decode error: {e}"));
        }
    };

    if let Some(code) = ds_body.get("code").and_then(|v| v.as_str()) {
        let message = ds_body
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("upstream error");
        recorder
            .record_error_response(request_id, session_id, &resp_headers, &raw)
            .await;
        return proxy_error(StatusCode::BAD_GATEWAY, &format!("{code}: {message}"));
    }

    debug!(
        "DashScope response: {}",
        serde_json::to_string_pretty(&ds_body).unwrap_or_default()
    );

    let openai_resp = dashscope_to_openai_response(&ds_body, remote_model);

    match format {
        OutputFormat::OpenAI => {
            recorder
                .record_response(request_id, session_id, &resp_headers, &openai_resp)
                .await;
            axum::Json(openai_resp).into_response()
        }
        OutputFormat::Anthropic => {
            let typed: ChatCompletionResponse = match serde_json::from_value(openai_resp) {
                Ok(v) => v,
                Err(e) => {
                    recorder
                        .record_error_response(request_id, session_id, &resp_headers, &raw)
                        .await;
                    return proxy_error(
                        StatusCode::BAD_GATEWAY,
                        &format!("response conversion error: {e}"),
                    );
                }
            };
            let anthropic_resp = convert::openai_to_anthropic_response(&typed);
            let resp_value = serde_json::to_value(&anthropic_resp).unwrap_or_default();
            recorder
                .record_response(request_id, session_id, &resp_headers, &resp_value)
                .await;
            axum::Json(anthropic_resp).into_response()
        }
    }
}

fn dashscope_to_openai_response(ds: &Value, model: &str) -> Value {
    let mut choices: Vec<Value> = Vec::new();

    if let Some(ds_choices) = ds
        .pointer("/output/choices")
        .and_then(|c| c.as_array())
    {
        for (index, choice) in ds_choices.iter().enumerate() {
            let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));

            let mut out_message = Map::new();
            out_message.insert(
                "role".to_string(),
                message
                    .get("role")
                    .cloned()
                    .unwrap_or_else(|| json!("assistant")),
            );
            out_message.insert("content".to_string(), normalize_content(message.get("content")));
            if let Some(reasoning) = message.get("reasoning_content") {
                if !reasoning.is_null() {
                    out_message.insert("reasoning_content".to_string(), reasoning.clone());
                }
            }
            if let Some(tool_calls) = message.get("tool_calls") {
                if !tool_calls.is_null() {
                    out_message.insert("tool_calls".to_string(), tool_calls.clone());
                }
            }

            choices.push(json!({
                "index": index,
                "message": out_message,
                "finish_reason": choice.get("finish_reason").cloned().unwrap_or(Value::Null),
            }));
        }
    }

    // Fallback for result_format=text responses.
    if choices.is_empty() {
        if let Some(text) = ds.pointer("/output/text").and_then(|v| v.as_str()) {
            choices.push(json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": ds
                    .pointer("/output/finish_reason")
                    .cloned()
                    .unwrap_or(Value::Null),
            }));
        }
    }

    let mut resp = json!({
        "id": ds
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("chatcmpl-proxy"),
        "object": "chat.completion",
        "created": now_unix(),
        "model": model,
        "choices": choices,
    });

    if let Some(usage) = ds.get("usage") {
        resp["usage"] = usage_to_openai(usage);
    }

    resp
}

fn normalize_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                Value::Null
            } else {
                Value::String(text)
            }
        }
        Some(Value::Null) | None => Value::Null,
        Some(other) => other.clone(),
    }
}

fn usage_to_openai(usage: &Value) -> Value {
    let input = usage
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = usage
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(input + output);
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": total,
    })
}

// ── Streaming chunk parsing ──

#[derive(Default)]
struct DashChunk {
    id: Option<String>,
    role: Option<String>,
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<Value>>,
    finish_reason: Option<String>,
    usage: Option<(u64, u64, u64)>,
    error: Option<String>,
}

fn parse_chunk(value: &Value) -> DashChunk {
    let mut chunk = DashChunk {
        id: value
            .get("request_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        ..Default::default()
    };

    if let Some(code) = value.get("code").and_then(|v| v.as_str()) {
        let message = value
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("upstream error");
        chunk.error = Some(format!("{code}: {message}"));
        return chunk;
    }

    if let Some(usage) = value.get("usage") {
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let total = usage
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(input + output);
        chunk.usage = Some((input, output, total));
    }

    let choice = value.pointer("/output/choices/0");
    if let Some(choice) = choice {
        chunk.finish_reason = choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| {
                value
                    .pointer("/output/finish_reason")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            });

        if let Some(message) = choice.get("message") {
            chunk.role = message
                .get("role")
                .and_then(|v| v.as_str())
                .map(String::from);
            chunk.content = message
                .get("content")
                .and_then(|v| v.as_str())
                .map(String::from);
            chunk.reasoning = message
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .map(String::from);
            chunk.tool_calls = message
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned();
        }
    }

    chunk
}

// ── Streaming: DashScope → OpenAI SSE ──

async fn stream_to_openai(
    resp: reqwest::Response,
    recorder: Recorder,
    request_id: String,
    resp_headers: String,
    session_id: Option<String>,
    remote_model: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut response_text = String::new();
        let mut reasoning_text = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut usage: Option<(u64, u64, u64)> = None;
        let mut id = String::new();
        let model = remote_model;
        let mut first = true;
        let mut has_data = false;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            buffer.extend_from_slice(&chunk);

            for line in drain_lines(&mut buffer) {
                let Some(data) = sse_data(&line) else {
                    continue;
                };
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let parsed = parse_chunk(&event);
                if let Some(err) = parsed.error {
                    recorder
                        .record_error_reason(&request_id, session_id.as_deref(), &err)
                        .await;
                    let payload = json!({"error": {"message": err, "type": "upstream_error"}});
                    let _ = tx.send(Ok(format!("data: {payload}\n\n"))).await;
                    continue;
                }

                has_data = true;
                if let Some(v) = parsed.id {
                    if id.is_empty() {
                        id = v;
                    }
                }
                if let Some(u) = parsed.usage {
                    usage = Some(u);
                }

                let mut delta = Map::new();
                if first {
                    let role = parsed.role.as_deref().unwrap_or("assistant");
                    delta.insert("role".to_string(), json!(role));
                    first = false;
                }
                if let Some(content) = &parsed.content {
                    if !content.is_empty() {
                        delta.insert("content".to_string(), json!(content));
                        response_text.push_str(content);
                    }
                }
                if let Some(reasoning) = &parsed.reasoning {
                    if !reasoning.is_empty() {
                        delta.insert("reasoning_content".to_string(), json!(reasoning));
                        reasoning_text.push_str(reasoning);
                    }
                }
                if let Some(tcs) = &parsed.tool_calls {
                    if !tcs.is_empty() {
                        delta.insert("tool_calls".to_string(), json!(tcs));
                        accumulate_tool_calls(&mut tool_calls, tcs);
                    }
                }
                if let Some(reason) = &parsed.finish_reason {
                    finish_reason = Some(reason.clone());
                }

                if !delta.is_empty() || parsed.finish_reason.is_some() {
                    let chunk_id = if id.is_empty() {
                        "chatcmpl-proxy"
                    } else {
                        &id
                    };
                    let out = json!({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": now_unix(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": parsed.finish_reason,
                        }],
                    });
                    let _ = tx.send(Ok(format!("data: {out}\n\n"))).await;
                }
            }
        }

        if let Some((input, output, total)) = usage {
            let chunk_id = if id.is_empty() {
                "chatcmpl-proxy"
            } else {
                &id
            };
            let out = json!({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": now_unix(),
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": input,
                    "completion_tokens": output,
                    "total_tokens": total,
                },
            });
            let _ = tx.send(Ok(format!("data: {out}\n\n"))).await;
        }
        let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;

        if has_data {
            let usage_value = usage.map(|(input, output, total)| {
                json!({
                    "prompt_tokens": input,
                    "completion_tokens": output,
                    "total_tokens": total,
                })
            });
            let assembled = json!({
                "id": if id.is_empty() { "chatcmpl-proxy" } else { &id },
                "object": "chat.completion",
                "created": now_unix(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": if response_text.is_empty() { Value::Null } else { json!(response_text) },
                        "reasoning_content": if reasoning_text.is_empty() { Value::Null } else { json!(reasoning_text) },
                        "tool_calls": if tool_calls.is_empty() { Value::Null } else { json!(tool_calls) },
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": usage_value,
            });
            recorder
                .record_response(&request_id, session_id.as_deref(), &resp_headers, &assembled)
                .await;
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

// ── Streaming: DashScope → Anthropic SSE ──

#[derive(Clone, Copy, PartialEq)]
enum BlockKind {
    Thinking,
    Text,
    ToolUse(usize),
}

async fn stream_to_anthropic(
    resp: reqwest::Response,
    recorder: Recorder,
    request_id: String,
    resp_headers: String,
    session_id: Option<String>,
    remote_model: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::io::Error>>(64);

    tokio::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();

        let mut id = String::new();
        let model = remote_model;
        let mut started = false;
        let mut next_index: u32 = 0;
        let mut current_block: Option<(BlockKind, u32)> = None;
        let mut tool_block_indices: HashMap<usize, u32> = HashMap::new();
        let mut response_text = String::new();
        let mut reasoning_text = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut stop_reason: Option<String> = None;
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut has_data = false;
        let mut failed = false;

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            buffer.extend_from_slice(&chunk);

            for line in drain_lines(&mut buffer) {
                let Some(data) = sse_data(&line) else {
                    continue;
                };
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let parsed = parse_chunk(&event);
                if let Some(err) = parsed.error {
                    recorder
                        .record_error_reason(&request_id, session_id.as_deref(), &err)
                        .await;
                    let payload = json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": err},
                    });
                    let _ = send_event(&tx, "error", &payload).await;
                    failed = true;
                    break;
                }

                has_data = true;
                if let Some(v) = parsed.id {
                    if id.is_empty() {
                        id = v;
                    }
                }
                if let Some((input, output, _)) = parsed.usage {
                    input_tokens = input;
                    output_tokens = output;
                }

                if !started {
                    started = true;
                    let message_id = if id.is_empty() { "msg_proxy" } else { &id };
                    let _ = send_event(
                        &tx,
                        "message_start",
                        &json!({
                            "type": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": model,
                                "stop_reason": null,
                                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                            }
                        }),
                    )
                    .await;
                }

                if let Some(reasoning) = parsed.reasoning.as_deref().filter(|r| !r.is_empty()) {
                    ensure_block(&tx, &mut current_block, &mut next_index, BlockKind::Thinking).await;
                    let index = current_block.map(|(_, i)| i).unwrap_or(0);
                    reasoning_text.push_str(reasoning);
                    let _ = send_event(
                        &tx,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        }),
                    )
                    .await;
                }

                if let Some(content) = parsed.content.as_deref().filter(|c| !c.is_empty()) {
                    ensure_block(&tx, &mut current_block, &mut next_index, BlockKind::Text).await;
                    let index = current_block.map(|(_, i)| i).unwrap_or(0);
                    response_text.push_str(content);
                    let _ = send_event(
                        &tx,
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": content},
                        }),
                    )
                    .await;
                }

                if let Some(tcs) = &parsed.tool_calls {
                    for tc in tcs {
                        let dash_index = tc
                            .get("index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;

                        let block_index = match tool_block_indices.get(&dash_index) {
                            Some(i) => *i,
                            None => {
                                close_block(&tx, &mut current_block).await;
                                let index = next_index;
                                next_index += 1;
                                tool_block_indices.insert(dash_index, index);
                                current_block = Some((BlockKind::ToolUse(dash_index), index));

                                let tool_id = tc
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("toolu_proxy");
                                let tool_name = tc
                                    .pointer("/function/name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let _ = send_event(
                                    &tx,
                                    "content_block_start",
                                    &json!({
                                        "type": "content_block_start",
                                        "index": index,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tool_id,
                                            "name": tool_name,
                                            "input": {},
                                        },
                                    }),
                                )
                                .await;
                                index
                            }
                        };

                        if let Some(arguments) = tc
                            .pointer("/function/arguments")
                            .and_then(|v| v.as_str())
                            .filter(|a| !a.is_empty())
                        {
                            let _ = send_event(
                                &tx,
                                "content_block_delta",
                                &json!({
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "input_json_delta", "partial_json": arguments},
                                }),
                            )
                            .await;
                        }
                    }
                    accumulate_tool_calls(&mut tool_calls, tcs);
                }

                if let Some(reason) = &parsed.finish_reason {
                    stop_reason = Some(map_stop_reason(reason).to_string());
                    if let Some((kind, index)) = current_block {
                        if !matches!(kind, BlockKind::ToolUse(_)) {
                            let _ = send_event(
                                &tx,
                                "content_block_stop",
                                &json!({"type": "content_block_stop", "index": index}),
                            )
                            .await;
                            current_block = None;
                        }
                    }
                }
            }

            if failed {
                break;
            }
        }

        if !failed {
            if !started {
                let message_id = if id.is_empty() { "msg_proxy" } else { &id };
                let _ = send_event(
                    &tx,
                    "message_start",
                    &json!({
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": null,
                            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                        }
                    }),
                )
                .await;
            }

            close_block(&tx, &mut current_block).await;

            let reason = stop_reason.unwrap_or_else(|| "end_turn".to_string());
            let _ = send_event(
                &tx,
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": reason, "stop_sequence": null},
                    "usage": {"output_tokens": output_tokens},
                }),
            )
            .await;
            let _ = send_event(&tx, "message_stop", &json!({"type": "message_stop"})).await;

            if has_data {
                let mut content_blocks: Vec<Value> = Vec::new();
                if !reasoning_text.is_empty() {
                    content_blocks.push(json!({
                        "type": "thinking",
                        "thinking": reasoning_text,
                        "signature": null,
                    }));
                }
                if !response_text.is_empty() {
                    content_blocks.push(json!({"type": "text", "text": response_text}));
                }
                for tc in &tool_calls {
                    let input = tc
                        .pointer("/function/arguments")
                        .and_then(|a| a.as_str())
                        .and_then(|a| serde_json::from_str::<Value>(a).ok())
                        .unwrap_or_else(|| json!({}));
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.get("id").cloned().unwrap_or_else(|| json!("toolu_proxy")),
                        "name": tc
                            .pointer("/function/name")
                            .cloned()
                            .unwrap_or_else(|| json!("")),
                        "input": input,
                    }));
                }

                let assembled = json!({
                    "id": if id.is_empty() { "msg_proxy" } else { &id },
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": content_blocks,
                    "stop_reason": reason,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                });
                recorder
                    .record_response(&request_id, session_id.as_deref(), &resp_headers, &assembled)
                    .await;
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

async fn ensure_block(
    tx: &tokio::sync::mpsc::Sender<Result<String, std::io::Error>>,
    current_block: &mut Option<(BlockKind, u32)>,
    next_index: &mut u32,
    kind: BlockKind,
) {
    if matches!(current_block, Some((k, _)) if *k == kind) {
        return;
    }
    close_block(tx, current_block).await;

    let index = *next_index;
    *next_index += 1;
    *current_block = Some((kind, index));

    let content_block = match kind {
        BlockKind::Thinking => json!({"type": "thinking", "thinking": ""}),
        BlockKind::Text => json!({"type": "text", "text": ""}),
        BlockKind::ToolUse(_) => return,
    };
    let _ = send_event(
        tx,
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        }),
    )
    .await;
}

async fn close_block(
    tx: &tokio::sync::mpsc::Sender<Result<String, std::io::Error>>,
    current_block: &mut Option<(BlockKind, u32)>,
) {
    if let Some((_, index)) = current_block.take() {
        let _ = send_event(
            tx,
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        )
        .await;
    }
}

// ── Helpers ──

fn drain_lines(buffer: &mut Vec<u8>) -> Vec<String> {
    let mut lines = Vec::new();
    while let Some(pos) = buffer.iter().position(|b| *b == b'\n') {
        let mut line: Vec<u8> = buffer.drain(..=pos).collect();
        line.pop();
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        lines.push(String::from_utf8_lossy(&line).trim().to_string());
    }
    lines
}

fn sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(|d| d.trim())
}

fn accumulate_tool_calls(tool_calls: &mut Vec<Value>, delta_tcs: &[Value]) {
    for tc in delta_tcs {
        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        while tool_calls.len() <= idx {
            tool_calls.push(json!({}));
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
                existing["function"] = json!({});
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

fn map_stop_reason(reason: &str) -> &str {
    match reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        other => other,
    }
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn send_event(
    tx: &tokio::sync::mpsc::Sender<Result<String, std::io::Error>>,
    event: &str,
    data: &Value,
) {
    let data_str = serde_json::to_string(data).unwrap_or_default();
    let _ = tx
        .send(Ok(format!("event: {event}\ndata: {data_str}\n\n")))
        .await;
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
