use super::{
    RunmatConfigDocument, RunmatConfigDocumentError, RunmatConfigFormat, RunmatConfigPatch,
};
use crate::desktop::{
    DesktopNotebookOnError, DesktopNotebookRerunAfterCancel, DesktopRunHistoryMode,
    DesktopRunLogMode,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::str::FromStr;

const LEGACY_KEYS: &[&str] = &[
    "artifact_root",
    "enable_gpu",
    "persist_agents",
    "background_runtime_diagnosis",
    "command_window_as_tab",
    "show_internal_artifacts",
    "notebook_persistence_mode",
    "notebook_interrupted_run_mode",
    "notebook_run_mode",
    "notebook_auto_restore_workspace",
    "run_clear_workspace_before_execution",
    "run_clear_figures_before_execution",
    "run_persist_trace",
    "run_persist_logs_mode",
    "runtime_error_reporting",
    "figure_scene_budget_bytes",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyDesktopMigration {
    pub source: String,
    pub changed: bool,
    pub removed_keys: Vec<String>,
}

pub fn migrate_legacy_desktop_config(
    source: &str,
    format: RunmatConfigFormat,
) -> Result<LegacyDesktopMigration, RunmatConfigDocumentError> {
    match format {
        RunmatConfigFormat::Toml => migrate_toml(source),
        RunmatConfigFormat::Json => migrate_json(source),
    }
}

/// Migrate a legacy Desktop document into an existing canonical project
/// document. Canonical destination keys win, unrelated destination sections
/// are preserved, and obsolete or preference-only legacy keys are removed
/// without being projected back into project configuration. A non-empty
/// legacy document reports `changed` even when the destination already
/// contains every value, so callers can complete the migration by deleting
/// the legacy file.
pub fn migrate_legacy_desktop_config_into(
    legacy_source: &str,
    destination_source: &str,
    format: RunmatConfigFormat,
) -> Result<LegacyDesktopMigration, RunmatConfigDocumentError> {
    migrate_legacy_desktop_config_between(legacy_source, format, destination_source, format)
}

/// Migrate a legacy Desktop document into a canonical document whose serialization format may
/// differ. This is the `.runmat` promotion boundary: legacy files are TOML-shaped, while an
/// existing canonical project may use either `runmat.toml` or `runmat.json`.
pub fn migrate_legacy_desktop_config_between(
    legacy_source: &str,
    legacy_format: RunmatConfigFormat,
    destination_source: &str,
    destination_format: RunmatConfigFormat,
) -> Result<LegacyDesktopMigration, RunmatConfigDocumentError> {
    let destination_source =
        if destination_format == RunmatConfigFormat::Json && destination_source.trim().is_empty() {
            "{}"
        } else {
            destination_source
        };
    let migrated = migrate_legacy_desktop_config(legacy_source, legacy_format)?;
    let fallback_source =
        convert_document_format(&migrated.source, legacy_format, destination_format)?;
    let (candidate, merged_missing_values) = match destination_format {
        RunmatConfigFormat::Toml => {
            let mut destination: toml_edit::DocumentMut =
                destination_source
                    .parse()
                    .map_err(|error: toml_edit::TomlError| {
                        RunmatConfigDocumentError::TomlParse(error.to_string())
                    })?;
            let fallback: toml_edit::DocumentMut =
                fallback_source
                    .parse()
                    .map_err(|error: toml_edit::TomlError| {
                        RunmatConfigDocumentError::TomlParse(error.to_string())
                    })?;
            let changed = merge_missing_toml(destination.as_table_mut(), fallback.as_table());
            (destination.to_string(), changed)
        }
        RunmatConfigFormat::Json => {
            let mut destination: Value = serde_json::from_str(destination_source)?;
            let fallback: Value = serde_json::from_str(&fallback_source)?;
            let changed = merge_missing_json(&mut destination, &fallback);
            (
                format!("{}\n", serde_json::to_string_pretty(&destination)?),
                changed,
            )
        }
    };
    let changed = migrated.changed || merged_missing_values || !legacy_source.trim().is_empty();
    let source = if changed {
        candidate
    } else {
        destination_source.to_string()
    };
    let merged = RunmatConfigDocument::parse(source, destination_format)?;
    Ok(LegacyDesktopMigration {
        source: merged.into_source(),
        changed,
        removed_keys: migrated.removed_keys,
    })
}

fn convert_document_format(
    source: &str,
    source_format: RunmatConfigFormat,
    destination_format: RunmatConfigFormat,
) -> Result<String, RunmatConfigDocumentError> {
    if source_format == destination_format {
        return Ok(source.to_string());
    }
    match (source_format, destination_format) {
        (RunmatConfigFormat::Toml, RunmatConfigFormat::Json) => {
            let value: toml::Value = toml::from_str(source)
                .map_err(|error| RunmatConfigDocumentError::TomlParse(error.to_string()))?;
            let value = serde_json::to_value(value)?;
            Ok(format!("{}\n", serde_json::to_string_pretty(&value)?))
        }
        (RunmatConfigFormat::Json, RunmatConfigFormat::Toml) => {
            let value: Value = serde_json::from_str(source)?;
            toml::to_string_pretty(&value)
                .map_err(|error| RunmatConfigDocumentError::TomlEdit(error.to_string()))
        }
        _ => unreachable!("all RunMat config format pairs are covered"),
    }
}

fn merge_missing_toml(destination: &mut toml_edit::Table, fallback: &toml_edit::Table) -> bool {
    let mut changed = false;
    for (key, fallback_item) in fallback.iter() {
        match destination.get_mut(key) {
            Some(destination_item) => {
                changed |= merge_missing_toml_item(destination_item, fallback_item);
            }
            None => {
                destination.insert(key, fallback_item.clone());
                changed = true;
            }
        }
    }
    changed
}

fn merge_missing_toml_item(destination: &mut toml_edit::Item, fallback: &toml_edit::Item) -> bool {
    match (destination, fallback) {
        (toml_edit::Item::Table(destination), toml_edit::Item::Table(fallback)) => {
            merge_missing_toml(destination, fallback)
        }
        (
            toml_edit::Item::Value(toml_edit::Value::InlineTable(destination)),
            toml_edit::Item::Value(toml_edit::Value::InlineTable(fallback)),
        ) => {
            let mut changed = false;
            for (key, fallback_value) in fallback.iter() {
                match destination.get_mut(key) {
                    Some(destination_value) => {
                        if let (
                            toml_edit::Value::InlineTable(destination),
                            toml_edit::Value::InlineTable(fallback),
                        ) = (destination_value, fallback_value)
                        {
                            changed |= merge_missing_inline_toml(destination, fallback);
                        }
                    }
                    None => {
                        destination.insert(key, fallback_value.clone());
                        changed = true;
                    }
                }
            }
            changed
        }
        _ => false,
    }
}

fn merge_missing_inline_toml(
    destination: &mut toml_edit::InlineTable,
    fallback: &toml_edit::InlineTable,
) -> bool {
    let mut changed = false;
    for (key, fallback_value) in fallback.iter() {
        match destination.get_mut(key) {
            Some(destination_value) => {
                if let (
                    toml_edit::Value::InlineTable(destination),
                    toml_edit::Value::InlineTable(fallback),
                ) = (destination_value, fallback_value)
                {
                    changed |= merge_missing_inline_toml(destination, fallback);
                }
            }
            None => {
                destination.insert(key, fallback_value.clone());
                changed = true;
            }
        }
    }
    changed
}

fn merge_missing_json(destination: &mut Value, fallback: &Value) -> bool {
    let (Some(destination), Some(fallback)) = (destination.as_object_mut(), fallback.as_object())
    else {
        return false;
    };
    let mut changed = false;
    for (key, fallback_value) in fallback {
        match destination.get_mut(key) {
            Some(destination_value) => {
                changed |= merge_missing_json(destination_value, fallback_value);
            }
            None => {
                destination.insert(key.clone(), fallback_value.clone());
                changed = true;
            }
        }
    }
    changed
}

fn migrate_toml(source: &str) -> Result<LegacyDesktopMigration, RunmatConfigDocumentError> {
    let newline = if source.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut section = String::new();
    let mut kept = Vec::new();
    let mut legacy = Map::new();
    let mut removed_keys = Vec::new();

    for line in source.lines() {
        if let Some(header) = table_header(line) {
            section = header;
            kept.push(line.to_string());
            continue;
        }
        if section == "desktop" {
            if let Some((key, raw_value)) = assignment(line) {
                if LEGACY_KEYS.contains(&key.as_str()) {
                    if legacy.contains_key(&key) {
                        return Err(RunmatConfigDocumentError::LegacyMigration(format!(
                            "legacy [desktop].{key} is declared more than once"
                        )));
                    }
                    legacy.insert(key.clone(), parse_legacy_toml_value(&key, raw_value)?);
                    removed_keys.push(key);
                    continue;
                }
            }
        }
        kept.push(line.to_string());
    }

    if legacy.is_empty() {
        return Ok(LegacyDesktopMigration {
            source: source.to_string(),
            changed: false,
            removed_keys,
        });
    }

    let mut cleaned = kept.join(newline);
    if source.ends_with('\n') {
        cleaned.push_str(newline);
    }
    let canonical: toml_edit::DocumentMut =
        cleaned.parse().map_err(|error: toml_edit::TomlError| {
            RunmatConfigDocumentError::TomlParse(error.to_string())
        })?;
    let patch = legacy_patch(&legacy, |path| toml_has_path(&canonical, path))?;
    let document = RunmatConfigDocument::parse(cleaned, RunmatConfigFormat::Toml)?;
    let migrated = document.patched(&patch)?;
    Ok(LegacyDesktopMigration {
        source: migrated.into_source(),
        changed: true,
        removed_keys,
    })
}

fn migrate_json(source: &str) -> Result<LegacyDesktopMigration, RunmatConfigDocumentError> {
    let mut document: Value = serde_json::from_str(source)?;
    let root = document
        .as_object_mut()
        .ok_or(RunmatConfigDocumentError::InvalidDocumentShape)?;
    let Some(desktop) = root.get_mut("desktop").and_then(Value::as_object_mut) else {
        return Ok(LegacyDesktopMigration {
            source: source.to_string(),
            changed: false,
            removed_keys: Vec::new(),
        });
    };

    let mut legacy = Map::new();
    let mut removed_keys = Vec::new();
    for key in LEGACY_KEYS {
        if let Some(value) = desktop.remove(*key) {
            legacy.insert((*key).to_string(), value);
            removed_keys.push((*key).to_string());
        }
    }
    if legacy.is_empty() {
        return Ok(LegacyDesktopMigration {
            source: source.to_string(),
            changed: false,
            removed_keys,
        });
    }

    let cleaned = format!("{}\n", serde_json::to_string_pretty(&document)?);
    let patch = legacy_patch(&legacy, |path| json_has_path(&document, path))?;
    let parsed = RunmatConfigDocument::parse(cleaned, RunmatConfigFormat::Json)?;
    let migrated = parsed.patched(&patch)?;
    Ok(LegacyDesktopMigration {
        source: migrated.into_source(),
        changed: true,
        removed_keys,
    })
}

fn legacy_patch(
    legacy: &Map<String, Value>,
    has_canonical: impl Fn(&[&str]) -> bool,
) -> Result<RunmatConfigPatch, RunmatConfigDocumentError> {
    let mut patch = RunmatConfigPatch::default();
    if !has_canonical(&["desktop", "artifacts", "root"]) {
        patch.desktop.artifacts.root = string_value(legacy, "artifact_root")?.map(PathBuf::from);
    }
    if !has_canonical(&["runtime", "accelerate", "enabled"]) {
        patch.runtime.accelerate_enabled = bool_value(legacy, "enable_gpu")?;
    }
    if !has_canonical(&["desktop", "run_history", "mode"]) {
        patch.desktop.run_history.mode = enum_string(legacy, "notebook_persistence_mode")?
            .map(|value| match value.as_str() {
                "off" => Ok(DesktopRunHistoryMode::Off),
                "smart" => Ok(DesktopRunHistoryMode::Budgeted),
                "full" => Ok(DesktopRunHistoryMode::Full),
                _ => invalid_value("notebook_persistence_mode", &value),
            })
            .transpose()?;
    }
    if !has_canonical(&["desktop", "notebook", "rerun_after_cancel"]) {
        patch.desktop.notebook.rerun_after_cancel =
            enum_string(legacy, "notebook_interrupted_run_mode")?
                .map(|value| match value.as_str() {
                    "remaining" => Ok(DesktopNotebookRerunAfterCancel::Remaining),
                    "full" => Ok(DesktopNotebookRerunAfterCancel::All),
                    _ => invalid_value("notebook_interrupted_run_mode", &value),
                })
                .transpose()?;
    }
    if !has_canonical(&["desktop", "notebook", "on_error"]) {
        patch.desktop.notebook.on_error = enum_string(legacy, "notebook_run_mode")?
            .map(|value| match value.as_str() {
                "stop_on_error" => Ok(DesktopNotebookOnError::Stop),
                "continue_on_error" => Ok(DesktopNotebookOnError::Continue),
                _ => invalid_value("notebook_run_mode", &value),
            })
            .transpose()?;
    }
    if !has_canonical(&["desktop", "script", "clear_workspace_before_run"]) {
        patch.desktop.script.clear_workspace_before_run =
            bool_value(legacy, "run_clear_workspace_before_execution")?;
    }
    if !has_canonical(&["desktop", "script", "clear_figures_before_run"]) {
        patch.desktop.script.clear_figures_before_run =
            bool_value(legacy, "run_clear_figures_before_execution")?;
    }
    if !has_canonical(&["desktop", "run_history", "trace"]) {
        patch.desktop.run_history.trace = bool_value(legacy, "run_persist_trace")?;
    }
    if !has_canonical(&["desktop", "run_history", "logs"]) {
        patch.desktop.run_history.logs = enum_string(legacy, "run_persist_logs_mode")?
            .map(|value| match value.as_str() {
                "off" => Ok(DesktopRunLogMode::Off),
                "errors" => Ok(DesktopRunLogMode::Errors),
                "all" => Ok(DesktopRunLogMode::All),
                _ => invalid_value("run_persist_logs_mode", &value),
            })
            .transpose()?;
    }
    if !has_canonical(&["runtime", "plotting", "export", "scene_budget_bytes"]) {
        patch.runtime.scene_budget_bytes = positive_integer(legacy, "figure_scene_budget_bytes")?;
    }
    Ok(patch)
}

fn table_header(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let without_comment = trimmed
        .split_once('#')
        .map_or(trimmed, |(value, _)| value)
        .trim();
    if !without_comment.starts_with('[')
        || !without_comment.ends_with(']')
        || without_comment.starts_with("[[")
    {
        return None;
    }
    Some(
        without_comment[1..without_comment.len() - 1]
            .trim()
            .to_ascii_lowercase(),
    )
}

fn assignment(line: &str) -> Option<(String, &str)> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let (key, value) = trimmed.split_once('=')?;
    let key = key.trim();
    if key.is_empty()
        || !key
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return None;
    }
    Some((key.to_ascii_lowercase(), value.trim()))
}

fn parse_legacy_toml_value(key: &str, raw: &str) -> Result<Value, RunmatConfigDocumentError> {
    let raw = strip_toml_comment(raw);
    if raw.is_empty() {
        return Err(RunmatConfigDocumentError::LegacyMigration(format!(
            "legacy [desktop].{key} has no value"
        )));
    }
    let wrapped = format!("value = {raw}");
    if let Ok(value) = toml::from_str::<toml::Value>(&wrapped) {
        return serde_json::to_value(value.get("value").cloned().ok_or_else(|| {
            RunmatConfigDocumentError::LegacyMigration(format!(
                "legacy [desktop].{key} has no value"
            ))
        })?)
        .map_err(Into::into);
    }
    Ok(Value::String(raw.trim_matches(['"', '\'']).to_string()))
}

fn strip_toml_comment(raw: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in raw.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if character == '\\' && quote == Some('"') {
            escaped = true;
            continue;
        }
        if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
            continue;
        }
        if character == '#' && quote.is_none() {
            return raw[..index].trim();
        }
    }
    raw.trim()
}

fn toml_has_path(document: &toml_edit::DocumentMut, path: &[&str]) -> bool {
    let mut item = document.as_item();
    for key in path {
        let Some(next) = item.get(*key) else {
            return false;
        };
        item = next;
    }
    true
}

fn json_has_path(document: &Value, path: &[&str]) -> bool {
    let mut value = document;
    for key in path {
        let Some(next) = value.get(*key) else {
            return false;
        };
        value = next;
    }
    true
}

fn string_value(
    legacy: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, RunmatConfigDocumentError> {
    let Some(value) = legacy.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| {
            RunmatConfigDocumentError::LegacyMigration(format!(
                "legacy [desktop].{key} must be text"
            ))
        })
}

fn enum_string(
    legacy: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>, RunmatConfigDocumentError> {
    string_value(legacy, key)
}

fn bool_value(
    legacy: &Map<String, Value>,
    key: &str,
) -> Result<Option<bool>, RunmatConfigDocumentError> {
    let Some(value) = legacy.get(key) else {
        return Ok(None);
    };
    if let Some(value) = value.as_bool() {
        return Ok(Some(value));
    }
    if let Some(value) = value.as_str() {
        return match value.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "1" => Ok(Some(true)),
            "false" | "no" | "off" | "0" => Ok(Some(false)),
            _ => invalid_value(key, value),
        };
    }
    Err(RunmatConfigDocumentError::LegacyMigration(format!(
        "legacy [desktop].{key} must be a boolean"
    )))
}

fn positive_integer(
    legacy: &Map<String, Value>,
    key: &str,
) -> Result<Option<usize>, RunmatConfigDocumentError> {
    let Some(value) = legacy.get(key) else {
        return Ok(None);
    };
    let integer = if let Some(value) = value.as_u64() {
        usize::try_from(value).ok()
    } else if let Some(value) = value.as_str() {
        usize::from_str(value).ok()
    } else {
        None
    };
    match integer {
        Some(value) if value > 0 => Ok(Some(value)),
        _ => Err(RunmatConfigDocumentError::LegacyMigration(format!(
            "legacy [desktop].{key} must be a positive integer"
        ))),
    }
}

fn invalid_value<T>(key: &str, value: &str) -> Result<T, RunmatConfigDocumentError> {
    Err(RunmatConfigDocumentError::LegacyMigration(format!(
        "legacy [desktop].{key} has unsupported value `{value}`"
    )))
}
