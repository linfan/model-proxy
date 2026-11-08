use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use tracing::debug;

use crate::proxy::anthropic;
use crate::proxy::dashscope;
use crate::proxy::openai as openai_proxy;
use crate::proxy::{proxy_error, ThinkingSettings};
use crate::record::generate_request_id;
use crate::server::AppState;
use crate::types::anthropic::MessagesRequest;
use crate::types::config::BackendType;
use crate::types::openai::*;

// ── POST /v1/chat/completions ──

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let request_id = generate_request_id();
    let session_id = extract_session_id(&headers);

    let model_name = match body.get("model").and_then(|v| v.as_str()) {
        Some(name) => name.to_string(),
        None => {
            let msg = "missing 'model' field";
            state.recorder.record_error_reason(&request_id, session_id.as_deref(), msg).await;
            return proxy_error(StatusCode::BAD_REQUEST, msg);
        }
    };

    let resolved = match state.config.resolve_model(&model_name) {
        Some(m) => m,
        None => {
            let msg = format!("unsupported model: {model_name}");
            state.recorder.record_error_reason(&request_id, session_id.as_deref(), &msg).await;
            return proxy_error(StatusCode::BAD_REQUEST, &msg);
        }
    };

    let thinking = ThinkingSettings::from_config(resolved);
    debug!("chat completions: model={model_name} backend={:?} thinking={:?}", resolved.backend_type, thinking);

    match resolved.backend_type {
        BackendType::OpenAI => {
            openai_proxy::forward_chat_completion(
                &state.http_client,
                resolved,
                body,
                &thinking,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
        BackendType::Anthropic => {
            let request: ChatCompletionRequest = match serde_json::from_value(body) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("invalid request: {e}");
                    state.recorder.record_error_reason(&request_id, session_id.as_deref(), &msg).await;
                    return proxy_error(StatusCode::BAD_REQUEST, &msg);
                }
            };
            anthropic::forward_chat_via_anthropic(
                &state.http_client,
                resolved,
                &request,
                &thinking,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
        BackendType::DashScope => {
            dashscope::forward_chat_completion(
                &state.http_client,
                resolved,
                body,
                &thinking,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
    }
}

// ── POST /v1/messages ──

pub async fn handle_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let request_id = generate_request_id();
    let session_id = extract_session_id(&headers);

    let model_name = match body.get("model").and_then(|v| v.as_str()) {
        Some(name) => name.to_string(),
        None => {
            let msg = "missing 'model' field";
            state.recorder.record_error_reason(&request_id, session_id.as_deref(), msg).await;
            return proxy_error(StatusCode::BAD_REQUEST, msg);
        }
    };

    let resolved = match state.config.resolve_model(&model_name) {
        Some(m) => m,
        None => {
            let msg = format!("unsupported model: {model_name}");
            state.recorder.record_error_reason(&request_id, session_id.as_deref(), &msg).await;
            return proxy_error(StatusCode::BAD_REQUEST, &msg);
        }
    };

    let thinking = ThinkingSettings::from_config(resolved);
    debug!("messages: model={model_name} backend={:?} thinking={:?}", resolved.backend_type, thinking);

    match resolved.backend_type {
        BackendType::Anthropic => {
            anthropic::forward_messages(
                &state.http_client,
                resolved,
                body,
                &thinking,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
        BackendType::OpenAI => {
            let request: MessagesRequest = match serde_json::from_value(body) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("invalid request: {e}");
                    state.recorder.record_error_reason(&request_id, session_id.as_deref(), &msg).await;
                    return proxy_error(StatusCode::BAD_REQUEST, &msg);
                }
            };
            anthropic::forward_messages_via_openai(
                &state.http_client,
                resolved,
                &request,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
        BackendType::DashScope => {
            let request: MessagesRequest = match serde_json::from_value(body) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("invalid request: {e}");
                    state.recorder.record_error_reason(&request_id, session_id.as_deref(), &msg).await;
                    return proxy_error(StatusCode::BAD_REQUEST, &msg);
                }
            };
            dashscope::forward_messages(
                &state.http_client,
                resolved,
                &request,
                &thinking,
                &state.recorder,
                &request_id,
                session_id.as_deref(),
            )
            .await
        }
    }
}

// ── GET /v1/models ──

pub async fn handle_list_models(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<ModelInfo> = state
        .config
        .all_model_names()
        .into_iter()
        .map(|(name, backend)| ModelInfo {
            id: name.to_string(),
            object: "model".to_string(),
            owned_by: match backend {
                BackendType::OpenAI => "openai".to_string(),
                BackendType::Anthropic => "anthropic".to_string(),
                BackendType::DashScope => "dashscope".to_string(),
            },
        })
        .collect();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}

fn extract_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}
