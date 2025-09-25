use anyhow::{anyhow, Result};

use crate::app::config::AppConfig;
use crate::builtin::inventory;
use crate::context::prompt;
use crate::context::snippets;
use crate::context::types::AuthoringContext;

/// Assemble a full authoring context for a builtin by name.
pub fn build_authoring_context(name: &str, config: &AppConfig) -> Result<AuthoringContext> {
    let manifest = inventory::collect_manifest()?;
    let record = manifest
        .builtins
        .into_iter()
        .find(|rec| rec.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| anyhow!("builtin '{name}' not found"))?;

    let prompt = prompt::render_prompt(&record, config);
    let source_paths = snippets::source_paths(&record, config)?;

    Ok(AuthoringContext {
        doc_markdown: record.doc_markdown.clone(),
        builtin: record,
        prompt,
        source_paths,
    })
}
