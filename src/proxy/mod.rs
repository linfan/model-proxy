pub mod anthropic;
pub mod convert;
pub mod openai;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::types::config::ResolvedModel;

#[derive(Debug, Clone)]
pub struct ThinkingSettings {
    pub enabled: bool,
    pub budget_tokens: Option<u32>,
}

impl ThinkingSettings {
    pub fn from_config(model: &ResolvedModel) -> Self {
        match model.entry.thinking.as_ref() {
            Some(cfg) => Self {
                enabled: cfg.enabled,
                budget_tokens: cfg.budget_tokens,
            },
            None => Self {
                enabled: false,
                budget_tokens: None,
            },
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
