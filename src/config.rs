use std::collections::HashMap;

use crate::types::config::*;

pub struct ConfigStore {
    pub config: AppConfig,
    models: HashMap<String, ResolvedModel>,
}

impl ConfigStore {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read config file {path}: {e}"))?;
        let config: AppConfig = serde_yaml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("failed to parse config YAML: {e}"))?;

        let mut models = HashMap::new();

        Self::register(&mut models, &config.backends.openai, BackendType::OpenAI)?;
        Self::register(&mut models, &config.backends.anthropic, BackendType::Anthropic)?;
        Self::register(&mut models, &config.backends.dashscope, BackendType::DashScope)?;

        if models.is_empty() {
            anyhow::bail!("no models configured");
        }

        Ok(Self { config, models })
    }

    fn register(
        models: &mut HashMap<String, ResolvedModel>,
        entries: &[ModelEntry],
        backend_type: BackendType,
    ) -> anyhow::Result<()> {
        for entry in entries {
            if models.contains_key(&entry.name) {
                anyhow::bail!("duplicate model name: {}", entry.name);
            }
            Self::validate_entry(entry)?;
            models.insert(
                entry.name.clone(),
                ResolvedModel {
                    backend_type,
                    entry: entry.clone(),
                },
            );
        }
        Ok(())
    }

    fn validate_entry(entry: &ModelEntry) -> anyhow::Result<()> {
        if entry.url.is_empty() {
            anyhow::bail!("model '{}': url is empty", entry.name);
        }
        if entry.api_token.is_empty() {
            anyhow::bail!("model '{}': api_token is empty", entry.name);
        }
        if entry.model.is_empty() {
            anyhow::bail!("model '{}': model is empty", entry.name);
        }
        Ok(())
    }

    pub fn resolve_model(&self, name: &str) -> Option<&ResolvedModel> {
        self.models.get(name)
    }

    pub fn all_model_names(&self) -> Vec<(&str, BackendType)> {
        self.models
            .iter()
            .map(|(name, rm)| (name.as_str(), rm.backend_type))
            .collect()
    }

    pub fn record_max_count(&self) -> usize {
        self.config.record.max_count
    }

    pub fn expand_record_path(&self) -> String {
        let path = &self.config.record.path;
        if path.starts_with('~') {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            path.replacen('~', &home, 1)
        } else {
            path.clone()
        }
    }
}
