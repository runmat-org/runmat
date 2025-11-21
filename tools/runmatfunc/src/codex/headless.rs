use std::path::PathBuf;

use anyhow::Result;

use crate::app::config::AppConfig;
use crate::codex::client::{self, CodexResponse};
use crate::codex::session;
use crate::codex::transcript::{PassRecord, Transcript};
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
    category: Option<String>,
    model: Option<String>,
    use_codex: bool,
) -> Result<HeadlessRunResult> {
    let authoring_ctx = gather::build_authoring_context(builtin, category.as_deref(), config)?;
    let resolved_model = model.clone().or_else(|| config.default_model.clone());

    let codex_available = client::is_available();
    let codex_summary = if use_codex && codex_available {
        session::run_authoring(&authoring_ctx, resolved_model.clone())?
    } else {
        None
    };

    let mut test_outcome = tests::run_builtin_tests(&authoring_ctx, config)?;

    // Multi-pass flow: packaging -> WGPU -> completion -> docs (only when codex used and tests ok)
    let mut passes: Vec<PassRecord> = Vec::new();
    if use_codex && codex_available && test_outcome.success {
        for pass in crate::context::passes::PASS_ORDER {
            let extra = (pass.build)(&authoring_ctx);
            if let Some(summary) = session::run_authoring_with_extra(
                &authoring_ctx,
                resolved_model.clone(),
                &extra,
                None,
            )? {
                test_outcome = tests::run_builtin_tests(&authoring_ctx, config)?;
                passes.push(PassRecord {
                    name: pass.name.to_string(),
                    codex_summary: Some(summary.summary),
                    passed: test_outcome.success,
                });
                if !test_outcome.success {
                    break;
                }
            }
        }
    }

    let transcript = Transcript::from_run(
        &authoring_ctx,
        resolved_model,
        codex_summary.as_ref().map(|resp| resp.summary.clone()),
        &test_outcome,
        passes,
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
