use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use tracing::debug;

use crate::proxy::anthropic;
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
    Json(body): Json<Value>,
) -> Response {
    let request_id = generate_request_id();

    let model_name = match body.get("model").and_then(|v| v.as_str()) {
        Some(name) => name.to_string(),
        None => return proxy_error(StatusCode::BAD_REQUEST, "missing 'model' field"),
    };

    let resolved = match state.config.resolve_model(&model_name) {
        Some(m) => m,
        None => {
            return proxy_error(
                StatusCode::BAD_REQUEST,
                &format!("unsupported model: {model_name}"),
            )
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
            )
            .await
        }
        BackendType::Anthropic => {
            let request: ChatCompletionRequest = match serde_json::from_value(body) {
                Ok(r) => r,
                Err(e) => return proxy_error(StatusCode::BAD_REQUEST, &format!("invalid request: {e}")),
            };
            anthropic::forward_chat_via_anthropic(
                &state.http_client,
                resolved,
                &request,
                &thinking,
                &state.recorder,
                &request_id,
            )
            .await
        }
    }
}

// ── POST /v1/messages ──

pub async fn handle_messages(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    let request_id = generate_request_id();

    let model_name = match body.get("model").and_then(|v| v.as_str()) {
        Some(name) => name.to_string(),
        None => return proxy_error(StatusCode::BAD_REQUEST, "missing 'model' field"),
    };

    let resolved = match state.config.resolve_model(&model_name) {
        Some(m) => m,
        None => {
            return proxy_error(
                StatusCode::BAD_REQUEST,
                &format!("unsupported model: {model_name}"),
            )
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
            )
            .await
        }
        BackendType::OpenAI => {
            let request: MessagesRequest = match serde_json::from_value(body) {
                Ok(r) => r,
                Err(e) => return proxy_error(StatusCode::BAD_REQUEST, &format!("invalid request: {e}")),
            };
            anthropic::forward_messages_via_openai(
                &state.http_client,
                resolved,
                &request,
                &state.recorder,
                &request_id,
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
            },
        })
        .collect();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: models,
    })
}
