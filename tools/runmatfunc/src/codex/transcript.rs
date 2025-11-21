use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::context::types::AuthoringContext;
use crate::workspace::tests::{TestCommandReport, TestOutcome};

#[derive(Debug, Serialize)]
pub struct PassRecord {
    pub name: String,
    pub codex_summary: Option<String>,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct Transcript {
    pub builtin: String,
    pub model: Option<String>,
    pub prompt: String,
    pub doc_markdown: Option<String>,
    pub source_paths: Vec<String>,
    pub codex_summary: Option<String>,
    pub passes: Vec<PassRecord>,
    pub tests: Vec<TestReportSummary>,
    pub test_log_dir: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct TestReportSummary {
    pub label: String,
    pub status: String,
    pub skipped: bool,
    pub note: Option<String>,
    pub stdout_log: Option<String>,
    pub stderr_log: Option<String>,
}

impl Transcript {
    pub fn from_run(
        ctx: &AuthoringContext,
        model: Option<String>,
        codex_summary: Option<String>,
        outcome: &TestOutcome,
        passes: Vec<PassRecord>,
    ) -> Self {
        let tests = outcome
            .reports
            .iter()
            .map(TestReportSummary::from_report)
            .collect();

        Self {
            builtin: ctx.builtin.name.clone(),
            model,
            prompt: ctx.prompt.clone(),
            doc_markdown: ctx.doc_markdown.clone(),
            source_paths: ctx
                .source_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
            codex_summary,
            passes,
            tests,
            test_log_dir: outcome.log_dir.display().to_string(),
            created_at: Utc::now(),
        }
    }

    pub fn write_to(&self, dir: &Path) -> Result<PathBuf> {
        fs::create_dir_all(dir)
            .with_context(|| format!("creating transcript dir {}", dir.display()))?;
        let timestamp = self.created_at.format("%Y%m%dT%H%M%SZ");
        let file_name = format!("{}_{}.json", timestamp, slugify(&self.builtin));
        let path = dir.join(file_name);
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)
            .with_context(|| format!("writing transcript {}", path.display()))?;
        Ok(path)
    }
}

impl TestReportSummary {
    fn from_report(report: &TestCommandReport) -> Self {
        let status = if report.skipped {
            "skipped"
        } else if report.success {
            "ok"
        } else {
            "failed"
        };
        Self {
            label: report.label.clone(),
            status: status.to_string(),
            skipped: report.skipped,
            note: report.note.clone(),
            stdout_log: report
                .stdout_path
                .as_ref()
                .map(|path| path.display().to_string()),
            stderr_log: report
                .stderr_path
                .as_ref()
                .map(|path| path.display().to_string()),
        }
    }
}

fn slugify(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
