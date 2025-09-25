use anyhow::{Context, Result};

use crate::codex::client::{default_client, CodexRequest};
use crate::context::types::AuthoringContext;

/// Run an authoring session through Codex. Returns optional summary text when the
/// client is available (None indicates stub/no client).
pub fn run_authoring(ctx: &AuthoringContext) -> Result<Option<String>> {
    let client = default_client()?;
    let request = CodexRequest {
        model: None,
        prompt: ctx.prompt.clone(),
        doc_markdown: ctx.doc_markdown.clone(),
        sources: ctx.source_paths.clone(),
    };

    match client.run(&request) {
        Ok(response) => Ok(Some(response.summary)),
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
