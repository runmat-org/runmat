use super::projection::RuntimeSection;
use super::{RunmatConfigDocumentError, RunmatConfigFormat};
use crate::runtime::RunMatRuntimeConfig;
use serde::Serialize;
use std::str::FromStr;

#[derive(Serialize)]
struct RuntimeDocument {
    runtime: RuntimeSection,
}

pub(super) fn replace_runtime(
    source: &str,
    format: RunmatConfigFormat,
    runtime: &RunMatRuntimeConfig,
) -> Result<String, RunmatConfigDocumentError> {
    let replacement = RuntimeDocument {
        runtime: RuntimeSection::from(runtime),
    };
    match format {
        RunmatConfigFormat::Toml => replace_toml(source, &replacement),
        RunmatConfigFormat::Json => replace_json(source, &replacement),
    }
}

fn replace_toml(
    source: &str,
    replacement: &RuntimeDocument,
) -> Result<String, RunmatConfigDocumentError> {
    let mut document = toml_edit::DocumentMut::from_str(source)
        .map_err(|error| RunmatConfigDocumentError::TomlEdit(error.to_string()))?;
    let serialized = toml_edit::ser::to_document(replacement)
        .map_err(|error| RunmatConfigDocumentError::TomlEdit(error.to_string()))?;
    let runtime = serialized
        .as_table()
        .get("runtime")
        .cloned()
        .ok_or_else(|| RunmatConfigDocumentError::TomlEdit("runtime table is missing".into()))?;
    document.as_table_mut().insert("runtime", runtime);
    Ok(document.to_string())
}

fn replace_json(
    source: &str,
    replacement: &RuntimeDocument,
) -> Result<String, RunmatConfigDocumentError> {
    let mut document: serde_json::Value = if source.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(source)?
    };
    let object = document
        .as_object_mut()
        .ok_or(RunmatConfigDocumentError::InvalidDocumentShape)?;
    let serialized = serde_json::to_value(replacement)?;
    let runtime = serialized
        .get("runtime")
        .cloned()
        .ok_or_else(|| RunmatConfigDocumentError::TomlEdit("runtime object is missing".into()))?;
    object.insert("runtime".into(), runtime);
    Ok(format!("{}\n", serde_json::to_string_pretty(&document)?))
}
