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

    pub async fn record_request(&self, request_id: &str, headers: &str, body: &Value) {
        if self.max_count == 0 {
            return;
        }
        let filename = self.filename(request_id, "sent");
        let body_str = serde_json::to_string_pretty(body).unwrap_or_default();
        let content = format!("{headers}\n\n{body_str}");
        if let Err(e) = fs::write(&filename, content).await {
            warn!("failed to record request: {e}");
        }
    }

    pub async fn record_response(&self, request_id: &str, headers: &str, body: &Value) {
        if self.max_count == 0 {
            return;
        }
        let filename = self.filename(request_id, "recv");
        let body_str = serde_json::to_string_pretty(body).unwrap_or_default();
        let content = format!("{headers}\n\n{body_str}");
        if let Err(e) = fs::write(&filename, content).await {
            warn!("failed to record response: {e}");
        }
    }

    fn filename(&self, request_id: &str, tag: &str) -> PathBuf {
        let now = chrono_format_now();
        self.dir.join(format!("{now}_{tag}_{request_id}.json"))
    }

    async fn cleanup(&self) {
        let mut entries = match fs::read_dir(&self.dir).await {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut files: Vec<(String, PathBuf)> = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let name = entry.file_name().to_string_lossy().to_string();
                files.push((name, path));
            }
        }

        // Group by request ID (timestamp_tag_requestid.json)
        let mut groups: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
        for (name, path) in &files {
            let parts: Vec<&str> = name.rsplitn(2, '_').collect();
            if parts.len() >= 2 {
                let _key = parts[1].to_string();
                // Use the full timestamp+requestid minus tag as group key
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

        debug!("cleaned up {delete_count} old record groups");
    }
}

fn chrono_format_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simple date calculation (good enough for filenames)
    let (year, month, day) = days_to_ymd(days);

    format!("{year:04}-{month:02}-{day:02}_{hours:02}-{minutes:02}-{seconds:02}")
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

pub fn generate_request_id() -> String {
    uuid::Uuid::new_v4().to_string()[..16].to_string()
}
