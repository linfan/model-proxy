use std::sync::Arc;

use axum::middleware as axum_middleware;
use axum::routing::{get, post};
use axum::Router;

use crate::config::ConfigStore;
use crate::handlers;
use crate::middleware;
use crate::record::Recorder;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ConfigStore>,
    pub http_client: reqwest::Client,
    pub recorder: Recorder,
}

pub fn build_router(state: AppState, url_prefix: &str) -> Router {
    let api_routes = Router::new()
        .route("/v1/chat/completions", post(handlers::handle_chat_completions))
        .route("/v1/messages", post(handlers::handle_messages))
        .route("/v1/models", get(handlers::handle_list_models))
        .layer(axum_middleware::from_fn(middleware::logging))
        .with_state(state);

    if url_prefix.is_empty() {
        api_routes
    } else {
        Router::new().nest(url_prefix, api_routes)
    }
}
