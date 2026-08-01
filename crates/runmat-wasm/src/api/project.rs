use std::path::Path;
use wasm_bindgen::prelude::*;

async fn freeze_handoff(
    source_path: &str,
) -> Result<Option<runmat_package::FrozenProjectHandoff>, JsValue> {
    let project = runmat_package::discover_frozen_project_from_async(
        Path::new(source_path),
        Default::default(),
    )
    .await
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    project
        .map(runmat_package::FrozenProjectHandoff::new)
        .map(|handoff| {
            handoff
                .validate()
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
            Ok(handoff)
        })
        .transpose()
}

#[wasm_bindgen(js_name = projectHandoff)]
pub async fn project_handoff(source_path: String) -> Result<JsValue, JsValue> {
    let Some(handoff) = freeze_handoff(&source_path).await? else {
        return Ok(JsValue::NULL);
    };
    serde_wasm_bindgen::to_value(&handoff).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = projectRevision)]
pub async fn project_revision(source_path: String) -> Result<JsValue, JsValue> {
    let Some(handoff) = freeze_handoff(&source_path).await? else {
        return Ok(JsValue::NULL);
    };
    serde_wasm_bindgen::to_value(&handoff.revision())
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen(js_name = validateProjectHandoff)]
pub fn validate_project_handoff(value: JsValue) -> Result<JsValue, JsValue> {
    let handoff: runmat_package::FrozenProjectHandoff = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    handoff
        .validate()
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&handoff.revision())
        .map_err(|error| JsValue::from_str(&error.to_string()))
}
