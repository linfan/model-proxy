pub mod anthropic;
pub mod convert;
pub mod openai;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::types::config::ResolvedModel;
use crate::types::openai::ThinkingOverride;

#[derive(Debug, Clone)]
pub struct ThinkingSettings {
    pub enabled: bool,
    pub budget_tokens: Option<u32>,
}

impl ThinkingSettings {
    pub fn resolve(model: &ResolvedModel, request_override: Option<&ThinkingOverride>) -> Self {
        let default_cfg = model.entry.thinking.as_ref();

        let enabled = request_override
            .and_then(|o| o.enabled)
            .or_else(|| default_cfg.map(|c| c.enabled))
            .unwrap_or(false);

        let budget_tokens = request_override
            .and_then(|o| o.budget_tokens)
            .or_else(|| default_cfg.and_then(|c| c.budget_tokens));

        Self {
            enabled,
            budget_tokens,
        }
    }
}

pub fn proxy_error(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": msg,
            "type": "proxy_error",
        }
    });
    (status, axum::Json(body)).into_response()
}
