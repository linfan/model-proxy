use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub record: RecordConfig,
    #[serde(default)]
    pub backends: BackendsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub url_prefix: String,
    #[serde(default)]
    pub debug: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8000,
            url_prefix: String::new(),
            debug: false,
        }
    }
}

fn default_port() -> u16 {
    8000
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RecordConfig {
    #[serde(default = "default_record_path")]
    pub path: String,
    #[serde(default = "default_max_count")]
    pub max_count: usize,
}

impl Default for RecordConfig {
    fn default() -> Self {
        Self {
            path: default_record_path(),
            max_count: 1000,
        }
    }
}

fn default_record_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.model-proxy/records")
}

fn default_max_count() -> usize {
    1000
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BackendsConfig {
    #[serde(default)]
    pub openai: Vec<ModelEntry>,
    #[serde(default)]
    pub anthropic: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub url: String,
    pub api_token: String,
    pub model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThinkingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    OpenAI,
    Anthropic,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub backend_type: BackendType,
    pub entry: ModelEntry,
}
