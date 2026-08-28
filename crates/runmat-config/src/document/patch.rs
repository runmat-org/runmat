use super::{RunmatConfigDocumentError, RunmatConfigFormat};
use crate::desktop::{
    DesktopNotebookOnError, DesktopNotebookRerunAfterCancel, DesktopRunHistoryMode,
    DesktopRunLogMode,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunmatConfigPatch {
    pub desktop: DesktopConfigPatch,
    pub runtime: RuntimeConfigPatch,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopConfigPatch {
    pub artifacts: DesktopArtifactsPatch,
    pub run_history: DesktopRunHistoryPatch,
    pub script: DesktopScriptPatch,
    pub notebook: DesktopNotebookPatch,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopArtifactsPatch {
    pub root: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopRunHistoryPatch {
    pub mode: Option<DesktopRunHistoryMode>,
    pub trace: Option<bool>,
    pub logs: Option<DesktopRunLogMode>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopScriptPatch {
    pub clear_workspace_before_run: Option<bool>,
    pub clear_figures_before_run: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DesktopNotebookPatch {
    pub on_error: Option<DesktopNotebookOnError>,
    pub rerun_after_cancel: Option<DesktopNotebookRerunAfterCancel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfigPatch {
    pub accelerate_enabled: Option<bool>,
    pub scene_budget_bytes: Option<usize>,
}

pub(super) fn apply_patch(
    source: &str,
    format: RunmatConfigFormat,
    patch: &RunmatConfigPatch,
) -> Result<String, RunmatConfigDocumentError> {
    match format {
        RunmatConfigFormat::Toml => patch_toml(source, patch),
        RunmatConfigFormat::Json => patch_json(source, patch),
    }
}

fn patch_toml(
    source: &str,
    patch: &RunmatConfigPatch,
) -> Result<String, RunmatConfigDocumentError> {
    let mut document = toml_edit::DocumentMut::from_str(source)
        .map_err(|error| RunmatConfigDocumentError::TomlEdit(error.to_string()))?;
    if let Some(value) = patch.desktop.artifacts.root.as_ref() {
        set_toml(
            document.as_table_mut(),
            &["desktop", "artifacts", "root"],
            toml_edit::value(value.to_string_lossy().as_ref()),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.mode {
        set_toml(
            document.as_table_mut(),
            &["desktop", "run_history", "mode"],
            toml_edit::value(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.trace {
        set_toml(
            document.as_table_mut(),
            &["desktop", "run_history", "trace"],
            toml_edit::value(value),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.logs {
        set_toml(
            document.as_table_mut(),
            &["desktop", "run_history", "logs"],
            toml_edit::value(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.script.clear_workspace_before_run {
        set_toml(
            document.as_table_mut(),
            &["desktop", "script", "clear_workspace_before_run"],
            toml_edit::value(value),
        )?;
    }
    if let Some(value) = patch.desktop.script.clear_figures_before_run {
        set_toml(
            document.as_table_mut(),
            &["desktop", "script", "clear_figures_before_run"],
            toml_edit::value(value),
        )?;
    }
    if let Some(value) = patch.desktop.notebook.on_error {
        set_toml(
            document.as_table_mut(),
            &["desktop", "notebook", "on_error"],
            toml_edit::value(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.notebook.rerun_after_cancel {
        set_toml(
            document.as_table_mut(),
            &["desktop", "notebook", "rerun_after_cancel"],
            toml_edit::value(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.runtime.accelerate_enabled {
        set_toml(
            document.as_table_mut(),
            &["runtime", "accelerate", "enabled"],
            toml_edit::value(value),
        )?;
    }
    if let Some(value) = patch.runtime.scene_budget_bytes {
        set_toml(
            document.as_table_mut(),
            &["runtime", "plotting", "export", "scene_budget_bytes"],
            toml_edit::value(i64::try_from(value).map_err(|_| {
                RunmatConfigDocumentError::TomlEdit(
                    "scene budget exceeds TOML integer range".into(),
                )
            })?),
        )?;
    }
    Ok(document.to_string())
}

fn set_toml(
    table: &mut toml_edit::Table,
    path: &[&str],
    value: toml_edit::Item,
) -> Result<(), RunmatConfigDocumentError> {
    let Some((key, parents)) = path.split_last() else {
        return Err(RunmatConfigDocumentError::TomlEdit(
            "empty config patch path".into(),
        ));
    };
    let mut current = table;
    for parent in parents {
        let item = current
            .entry(parent)
            .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
        current = item.as_table_mut().ok_or_else(|| {
            RunmatConfigDocumentError::TomlEdit(format!(
                "cannot create `{}` because `{parent}` is not a table",
                path.join(".")
            ))
        })?;
    }
    current.insert(key, value);
    Ok(())
}

fn patch_json(
    source: &str,
    patch: &RunmatConfigPatch,
) -> Result<String, RunmatConfigDocumentError> {
    let mut document: serde_json::Value = if source.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(source)?
    };
    if !document.is_object() {
        return Err(RunmatConfigDocumentError::InvalidDocumentShape);
    }
    if let Some(value) = patch.desktop.artifacts.root.as_ref() {
        set_json(
            &mut document,
            &["desktop", "artifacts", "root"],
            serde_json::Value::String(value.to_string_lossy().into_owned()),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.mode {
        set_json(
            &mut document,
            &["desktop", "run_history", "mode"],
            serde_json::Value::String(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.trace {
        set_json(
            &mut document,
            &["desktop", "run_history", "trace"],
            value.into(),
        )?;
    }
    if let Some(value) = patch.desktop.run_history.logs {
        set_json(
            &mut document,
            &["desktop", "run_history", "logs"],
            serde_json::Value::String(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.script.clear_workspace_before_run {
        set_json(
            &mut document,
            &["desktop", "script", "clear_workspace_before_run"],
            value.into(),
        )?;
    }
    if let Some(value) = patch.desktop.script.clear_figures_before_run {
        set_json(
            &mut document,
            &["desktop", "script", "clear_figures_before_run"],
            value.into(),
        )?;
    }
    if let Some(value) = patch.desktop.notebook.on_error {
        set_json(
            &mut document,
            &["desktop", "notebook", "on_error"],
            serde_json::Value::String(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.desktop.notebook.rerun_after_cancel {
        set_json(
            &mut document,
            &["desktop", "notebook", "rerun_after_cancel"],
            serde_json::Value::String(enum_value(value)?),
        )?;
    }
    if let Some(value) = patch.runtime.accelerate_enabled {
        set_json(
            &mut document,
            &["runtime", "accelerate", "enabled"],
            value.into(),
        )?;
    }
    if let Some(value) = patch.runtime.scene_budget_bytes {
        set_json(
            &mut document,
            &["runtime", "plotting", "export", "scene_budget_bytes"],
            serde_json::Value::from(u64::try_from(value).map_err(|_| {
                RunmatConfigDocumentError::TomlEdit(
                    "scene budget exceeds JSON integer range".into(),
                )
            })?),
        )?;
    }
    Ok(format!("{}\n", serde_json::to_string_pretty(&document)?))
}

fn set_json(
    document: &mut serde_json::Value,
    path: &[&str],
    value: serde_json::Value,
) -> Result<(), RunmatConfigDocumentError> {
    let Some((key, parents)) = path.split_last() else {
        return Err(RunmatConfigDocumentError::TomlEdit(
            "empty config patch path".into(),
        ));
    };
    let mut current = document;
    for parent in parents {
        let object = current.as_object_mut().ok_or_else(|| {
            RunmatConfigDocumentError::TomlEdit(format!(
                "cannot create `{}` because `{parent}` is not an object",
                path.join(".")
            ))
        })?;
        current = object
            .entry((*parent).to_string())
            .or_insert_with(|| serde_json::json!({}));
    }
    current
        .as_object_mut()
        .ok_or_else(|| {
            RunmatConfigDocumentError::TomlEdit(format!(
                "cannot set `{}` because its parent is not an object",
                path.join(".")
            ))
        })?
        .insert((*key).to_string(), value);
    Ok(())
}

fn enum_value<T: Serialize>(value: T) -> Result<String, RunmatConfigDocumentError> {
    serde_json::to_value(value)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| RunmatConfigDocumentError::TomlEdit("enum did not serialize as text".into()))
}
