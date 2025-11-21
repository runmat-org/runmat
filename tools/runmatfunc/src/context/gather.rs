use anyhow::Result;

use crate::app::config::AppConfig;
use crate::builtin::inventory;
use crate::builtin::metadata::BuiltinRecord;
use crate::context::prompt;
use crate::context::snippets;
use crate::context::types::AuthoringContext;

/// Assemble a full authoring context for a builtin by name.
pub fn build_authoring_context(
    name: &str,
    category_override: Option<&str>,
    config: &AppConfig,
) -> Result<AuthoringContext> {
    let manifest = inventory::collect_manifest()?;
    let mut record = manifest
        .builtins
        .into_iter()
        .find(|rec| rec.name.eq_ignore_ascii_case(name))
        .unwrap_or_else(|| placeholder_record(name, category_override));

    if let Some(category) = category_override {
        record.category = Some(category.to_string());
    }

    let prompt = prompt::render_prompt(&record, config);
    let source_paths = snippets::source_paths(&record, config)?;

    Ok(AuthoringContext {
        doc_markdown: record.doc_markdown.clone(),
        builtin: record,
        prompt,
        source_paths,
    })
}

fn placeholder_record(name: &str, category: Option<&str>) -> BuiltinRecord {
    BuiltinRecord {
        name: name.to_string(),
        category: category.map(|c| c.to_string()),
        summary: None,
        keywords: Vec::new(),
        accel_tags: Vec::new(),
        is_sink: false,
        doc_markdown: None,
        param_types: Vec::new(),
        return_type: "Value".to_string(),
    }
}
