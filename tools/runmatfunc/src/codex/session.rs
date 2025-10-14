use anyhow::{Context, Result};

use crate::codex::client::{default_client, CodexRequest, CodexResponse};
use crate::context::types::AuthoringContext;

/// Run an authoring session through Codex. Returns optional summary text when the
/// client is available (None indicates stub/no client).
pub fn run_authoring(
    ctx: &AuthoringContext,
    model: Option<String>,
) -> Result<Option<CodexResponse>> {
    let client = default_client()?;
    let request = CodexRequest {
        model,
        prompt: ctx.prompt.clone(),
        doc_markdown: ctx.doc_markdown.clone(),
        sources: ctx.source_paths.clone(),
    };

    match client.run(&request) {
        Ok(response) => Ok(Some(response)),
        Err(err)
            if err
                .to_string()
                .contains("requires the 'embedded-codex' feature") =>
        {
            Ok(None)
        }
        Err(err) => Err(err).with_context(|| "codex authoring session failed"),
    }
}

#[cfg(all(test, feature = "embedded-codex"))]
mod embedded_tests {
    use super::*;
    use crate::builtin::metadata::BuiltinRecord;

    #[test]
    fn fixture_summary_is_returned() {
        let _fixture_flag =
            crate::codex::client::EnvVarGuard::set("RUNMATFUNC_USE_CODEX_FIXTURE", "1");
        let record = BuiltinRecord {
            name: "fixture".to_string(),
            category: Some("tests".to_string()),
            summary: Some("Fixture builtin".to_string()),
            keywords: Vec::new(),
            accel_tags: Vec::new(),
            is_sink: false,
            doc_markdown: Some("RunMat Codex fixture documentation".to_string()),
            param_types: Vec::new(),
            return_type: "Value".to_string(),
        };

        let ctx = AuthoringContext {
            builtin: record,
            prompt: "Summarize the RunMat fixture".to_string(),
            doc_markdown: Some("RunMat Codex fixture documentation".to_string()),
            source_paths: Vec::new(),
        };

        let result = match run_authoring(&ctx, Some("fixture-model".to_string())) {
            Ok(Some(summary)) => summary,
            Ok(None) => {
                eprintln!("codex CLI unavailable; skipping session fixture test");
                return;
            }
            Err(err) => {
                eprintln!("codex CLI unavailable; skipping session fixture test: {err:?}");
                return;
            }
        };
        let summary = result;
        assert!(
            !summary.summary.trim().is_empty(),
            "fixture should yield non-empty Codex response"
        );
    }
}

#[cfg(all(test, not(feature = "embedded-codex")))]
mod stub_tests {
    use super::*;
    use crate::builtin::metadata::BuiltinRecord;

    #[test]
    fn stub_client_returns_none() {
        let record = BuiltinRecord {
            name: "fixture".to_string(),
            category: None,
            summary: None,
            keywords: Vec::new(),
            accel_tags: Vec::new(),
            is_sink: false,
            doc_markdown: None,
            param_types: Vec::new(),
            return_type: "Value".to_string(),
        };

        let ctx = AuthoringContext {
            builtin: record,
            prompt: "Test".to_string(),
            doc_markdown: None,
            source_paths: Vec::new(),
        };

        let result = run_authoring(&ctx, None).expect("stub client should succeed");
        assert!(result.is_none());
    }
}
