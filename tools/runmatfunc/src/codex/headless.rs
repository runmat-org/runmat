use std::path::PathBuf;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::codex::client::{self, CodexResponse};
use crate::codex::session;
use crate::codex::transcript::Transcript;
use crate::context::gather;
use crate::workspace::tests::{self, TestOutcome};

pub struct HeadlessRunResult {
    pub transcript_path: PathBuf,
    pub transcript: Transcript,
    pub test_outcome: TestOutcome,
    pub codex_summary: Option<CodexResponse>,
}

pub fn run_builtin_headless(
    config: &AppConfig,
    builtin: &str,
    model: Option<String>,
    use_codex: bool,
) -> Result<HeadlessRunResult> {
    let authoring_ctx = gather::build_authoring_context(builtin, None, config)?;
    let resolved_model = model.clone().or_else(|| config.default_model.clone());

    let codex_available = client::is_available();
    let codex_summary = if use_codex && codex_available {
        session::run_authoring(&authoring_ctx, resolved_model.clone())?
    } else {
        None
    };

    let test_outcome = tests::run_builtin_tests(&authoring_ctx, config)?;

    let transcript = Transcript::from_run(
        &authoring_ctx,
        resolved_model,
        codex_summary.as_ref().map(|resp| resp.summary.clone()),
        &test_outcome,
    );

    let transcript_dir = config.transcripts_dir();
    let transcript_path = transcript.write_to(&transcript_dir)?;

    Ok(HeadlessRunResult {
        transcript_path,
        transcript,
        test_outcome,
        codex_summary,
    })
}
