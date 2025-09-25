use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::app::config::AppConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueEntry {
    pub builtin: String,
    pub model: Option<String>,
    #[serde(default)]
    pub use_codex: bool,
    pub enqueued_at: String,
}

impl QueueEntry {
    pub fn new(builtin: String, model: Option<String>, use_codex: bool) -> Self {
        Self {
            builtin,
            model,
            use_codex,
            enqueued_at: Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobQueue {
    entries: VecDeque<QueueEntry>,
}

impl JobQueue {
    pub fn push(&mut self, entry: QueueEntry) {
        self.entries.push_back(entry);
    }

    pub fn pop_front(&mut self) -> Option<QueueEntry> {
        self.entries.pop_front()
    }

    pub fn iter(&self) -> impl Iterator<Item = &QueueEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn load(config: &AppConfig) -> Result<JobQueue> {
    let path = config.queue_file_path();
    if !path.exists() {
        return Ok(JobQueue::default());
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("reading queue file {}", path.display()))?;
    let queue: JobQueue = serde_json::from_str(&content)
        .with_context(|| format!("parsing queue file {}", path.display()))?;
    Ok(queue)
}

pub fn save(config: &AppConfig, queue: &JobQueue) -> Result<()> {
    let path = config.queue_file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating queue dir {}", parent.display()))?;
    }
    let content = serde_json::to_string_pretty(queue)?;
    fs::write(&path, content).with_context(|| format!("writing queue file {}", path.display()))
}

pub fn add_entry(config: &AppConfig, entry: QueueEntry) -> Result<JobQueue> {
    let mut queue = load(config)?;
    queue.push(entry);
    save(config, &queue)?;
    Ok(queue)
}

pub fn clear(config: &AppConfig) -> Result<()> {
    let empty = JobQueue::default();
    save(config, &empty)
}

pub fn queue_path(config: &AppConfig) -> PathBuf {
    config.queue_file_path()
}
