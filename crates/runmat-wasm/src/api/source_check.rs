//! Ordinary script/function/project frontend, not a JavaScript shadow checker.
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Uses the currently installed host filesystem and optional immutable input
/// manifest. It does not create a session, collector, GPU or foreign runtime.
#[wasm_bindgen(js_name = checkSource)]
pub async fn check_source(request: JsValue) -> Result<JsValue, JsValue> {
    let request: runmat_core::source_check::SourceCheckRequest =
        serde_wasm_bindgen::from_value(request).map_err(|error| {
            JsValue::from_str(&format!("Invalid source check request: {error}"))
        })?;
    runmat_runtime::builtins::wasm_registry::register_all();
    let report = runmat_core::source_check::check_source(request)
        .await
        .map_err(|error| JsValue::from_str(&error))?;
    report
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|error| {
            JsValue::from_str(&format!("Cannot serialize source check report: {error}"))
        })
}
