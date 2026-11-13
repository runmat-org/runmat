//! Canonical user/project settings boundary; no browser filesystem assumptions.
use runmat_config::document::RunmatConfigFormat;
use runmat_config::runtime::{HostRuntimeSettings, UserSettingsDocument};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = resolveRuntimeSettings)]
pub fn resolve_runtime_settings(layers: JsValue) -> Result<JsValue, JsValue> {
    let layers: HostRuntimeSettings = serde_wasm_bindgen::from_value(layers).map_err(error)?;
    let resolved = layers.resolve().map_err(error)?;
    resolved
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(error)
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "camelCase", deny_unknown_fields)]
enum UserSettingsEdit {
    Set {
        path: Vec<String>,
        value: serde_json::Value,
    },
    Reset {
        path: Vec<String>,
    },
}

#[wasm_bindgen(js_name = editUserRuntimeSettings)]
pub fn edit_user_runtime_settings(source: String, edits: JsValue) -> Result<String, JsValue> {
    let edits: Vec<UserSettingsEdit> = serde_wasm_bindgen::from_value(edits).map_err(error)?;
    let mut document =
        UserSettingsDocument::parse(source, RunmatConfigFormat::Json).map_err(error)?;
    for edit in edits {
        match edit {
            UserSettingsEdit::Set { path, value } => {
                document.set(&path.iter().map(String::as_str).collect::<Vec<_>>(), value)
            }
            UserSettingsEdit::Reset { path } => {
                document.reset(&path.iter().map(String::as_str).collect::<Vec<_>>())
            }
        }
        .map_err(error)?;
    }
    Ok(document.source().into())
}

fn error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
