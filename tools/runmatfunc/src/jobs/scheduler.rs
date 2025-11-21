use std::path::PathBuf;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::codex::headless::{self, HeadlessRunResult};
use crate::jobs::queue::{JobQueue, QueueEntry};

#[derive(Debug)]
pub struct JobRunOutcome {
    pub entry: QueueEntry,
    pub transcript_path: Option<PathBuf>,
    pub success: bool,
    pub error: Option<String>,
    pub codex_summary: Option<String>,
}

pub fn run_queue(
    config: &AppConfig,
    queue: &mut JobQueue,
    max_jobs: Option<usize>,
) -> Vec<JobRunOutcome> {
    let limit = max_jobs.unwrap_or_else(|| queue.len());
    let mut outcomes = Vec::new();

    for _ in 0..limit {
        let Some(entry) = queue.pop_front() else {
            break;
        };
        match run_entry(config, &entry) {
            Ok(result) => {
                let HeadlessRunResult {
                    transcript_path,
                    transcript: _,
                    test_outcome,
                    codex_summary,
                } = result;
                outcomes.push(JobRunOutcome {
                    codex_summary: codex_summary.map(|resp| resp.summary),
                    entry,
                    transcript_path: Some(transcript_path),
                    success: test_outcome.success,
                    error: None,
                })
            }
            Err(err) => outcomes.push(JobRunOutcome {
                entry,
                transcript_path: None,
                success: false,
                error: Some(err.to_string()),
                codex_summary: None,
            }),
        }
    }

    outcomes
}

fn run_entry(config: &AppConfig, entry: &QueueEntry) -> Result<HeadlessRunResult> {
    headless::run_builtin_headless(
        config,
        &entry.builtin,
        entry.category.clone(),
        entry.model.clone(),
        entry.use_codex,
    )
}
