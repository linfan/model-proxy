mod config;
mod handlers;
mod middleware;
mod proxy;
mod record;
mod server;
mod types;

use std::sync::Arc;

use clap::Parser;
use tracing::info;

#[derive(Parser)]
#[command(name = "model-proxy", about = "Model proxy server for OpenAI and Anthropic APIs")]
struct Cli {
    /// Enable debug mode with verbose logging
    #[arg(short, long)]
    debug: bool,

    /// Port to listen on
    #[arg(short, long, default_value = "8000")]
    port: u16,

    /// Path to YAML config file
    #[arg(short, long, default_value = "~/.model-proxy/config.yaml")]
    config: String,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let log_level = if cli.debug { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .init();

    let config_path = expand_tilde(&cli.config);
    let store = match config::ConfigStore::load(&config_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    };

    let port = if cli.port != 8000 {
        cli.port
    } else {
        store.config.server.port
    };
    let url_prefix = normalize_prefix(&store.config.server.url_prefix);
    let record_path = store.expand_record_path();
    let max_count = store.record_max_count();

    let recorder = record::Recorder::new(&record_path, max_count).await;
    recorder.start_cleanup_task();

    let state = server::AppState {
        config: Arc::new(store),
        http_client: reqwest::Client::new(),
        recorder,
    };

    let app = server::build_router(state, &url_prefix);

    let addr = format!("0.0.0.0:{port}");
    if url_prefix.is_empty() {
        info!("Starting server on {addr}");
    } else {
        info!("Starting server on {addr} with URL prefix: {url_prefix}");
    }

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn expand_tilde(path: &str) -> String {
    if path.starts_with('~') {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        path.replacen('~', &home, 1)
    } else {
        path.to_string()
    }
}

fn normalize_prefix(prefix: &str) -> String {
    if prefix.is_empty() {
        return String::new();
    }
    let mut p = prefix.to_string();
    if !p.starts_with('/') {
        p = format!("/{p}");
    }
    p = p.trim_end_matches('/').to_string();
    p
}
