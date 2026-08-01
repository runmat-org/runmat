use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = buildGitSnapshot)]
pub fn build_git_snapshot(
    repository: &str,
    subdir: &str,
    value: JsValue,
) -> Result<JsValue, JsValue> {
    let inventory: runmat_package_cache::GitTreeInventory =
        serde_wasm_bindgen::from_value(value)
            .map_err(|error| JsValue::from_str(&format!("Git inventory parse failed: {error}")))?;
    let snapshot = inventory
        .into_snapshot(
            repository,
            subdir,
            runmat_package_cache::ArchiveLimits::default(),
        )
        .map_err(|error| JsValue::from_str(&format!("Git inventory validation failed: {error}")))?;
    serde_wasm_bindgen::to_value(&snapshot)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot serialization failed: {error}")))
}

#[wasm_bindgen(js_name = validateGitSnapshot)]
pub fn validate_git_snapshot(value: JsValue) -> Result<JsValue, JsValue> {
    let snapshot: runmat_package_cache::GitSnapshot = serde_wasm_bindgen::from_value(value)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot parse failed: {error}")))?;
    snapshot
        .validate()
        .map_err(|error| JsValue::from_str(&format!("Git snapshot validation failed: {error}")))?;
    serde_wasm_bindgen::to_value(&snapshot)
        .map_err(|error| JsValue::from_str(&format!("Git snapshot serialization failed: {error}")))
}
