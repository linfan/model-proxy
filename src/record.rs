use std::path::PathBuf;
use serde_json::Value;
use tokio::fs;
use tracing::{debug, warn};

#[derive(Clone)]
pub struct Recorder {
    dir: PathBuf,
    max_count: usize,
}

impl Recorder {
    pub async fn new(dir: &str, max_count: usize) -> Self {
        let path = PathBuf::from(dir);
        if max_count > 0 {
            if let Err(e) = fs::create_dir_all(&path).await {
                warn!("failed to create record directory {dir}: {e}");
            }
        }
        Self {
            dir: path,
            max_count,
        }
    }

    pub fn start_cleanup_task(&self) {
        if self.max_count == 0 {
            return;
        }
        let recorder = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(600));
            loop {
                interval.tick().await;
                recorder.cleanup().await;
            }
        });
    }

    pub async fn record_request(&self, request_id: &str, session_id: Option<&str>, headers: &str, body: &Value) {
        if self.max_count == 0 {
            return;
        }
        let filename = self.filename(request_id, "sent", session_id).await;
        let body_str = serde_json::to_string_pretty(body).unwrap_or_default();
        let content = format!("{headers}\n\n{body_str}");
        if let Err(e) = fs::write(&filename, content).await {
            warn!("failed to record request: {e}");
        }
    }

    pub async fn record_response(&self, request_id: &str, session_id: Option<&str>, headers: &str, body: &Value) {
        if self.max_count == 0 {
            return;
        }
        let filename = self.filename(request_id, "recv", session_id).await;
        let body_str = serde_json::to_string_pretty(body).unwrap_or_default();
        let content = format!("{headers}\n\n{body_str}");
        if let Err(e) = fs::write(&filename, content).await {
            warn!("failed to record response: {e}");
        }
    }

    async fn filename(&self, request_id: &str, tag: &str, session_id: Option<&str>) -> PathBuf {
        let now = chrono_format_now();
        let date = date_today();
        let subdir = match session_id {
            Some(id) => self.dir.join(format!("{date}_{id}")),
            None => self.dir.join(format!("{date}_unknown-session")),
        };
        let _ = fs::create_dir_all(&subdir).await;
        subdir.join(format!("{now}_{tag}_{request_id}.json"))
    }

    async fn cleanup(&self) {
        let mut top_entries = match fs::read_dir(&self.dir).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut subdirs: Vec<PathBuf> = Vec::new();
        while let Ok(Some(entry)) = top_entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                subdirs.push(path);
            }
        }

        let mut files: Vec<(String, PathBuf)> = Vec::new();
        for subdir in &subdirs {
            let mut entries = match fs::read_dir(subdir).await {
                Ok(e) => e,
                Err(_) => continue,
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    let name = entry.file_name().to_string_lossy().to_string();
                    files.push((name, path));
                }
            }
        }

        // Group by request ID (timestamp_tag_requestid.json)
        let mut groups: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
        for (name, path) in &files {
            let parts: Vec<&str> = name.rsplitn(2, '_').collect();
            if parts.len() >= 2 {
                let group_key = name
                    .strip_suffix(".json")
                    .unwrap_or(name)
                    .rsplitn(3, '_')
                    .last()
                    .unwrap_or("")
                    .to_string();
                groups.entry(group_key).or_default().push(path.clone());
            }
        }

        if groups.len() <= self.max_count {
            return;
        }

        let mut group_keys: Vec<String> = groups.keys().cloned().collect();
        group_keys.sort();

        let delete_count = groups.len() - self.max_count;
        for key in group_keys.iter().take(delete_count) {
            if let Some(paths) = groups.get(key) {
                for path in paths {
                    let _ = fs::remove_file(path).await;
                }
            }
        }

        // Remove empty subdirectories
        for subdir in &subdirs {
            let mut entries = match fs::read_dir(subdir).await {
                Ok(e) => e,
                Err(_) => continue,
            };
            if entries.next_entry().await.ok().flatten().is_none() {
                let _ = fs::remove_dir(subdir).await;
            }
        }

        debug!("cleaned up {delete_count} old record groups");
    }
}

fn local_now() -> libc::tm {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r only writes to the provided tm buffer.
    unsafe {
        libc::localtime_r(&secs, &mut tm);
    }
    tm
}

fn date_today() -> String {
    let tm = local_now();
    format!(
        "{:04}{:02}{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday
    )
}

fn chrono_format_now() -> String {
    let tm = local_now();
    format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}

pub fn generate_request_id() -> String {
    uuid::Uuid::new_v4().to_string()[..16].to_string()
}
