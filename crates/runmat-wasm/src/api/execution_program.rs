use wasm_bindgen::prelude::*;

use runmat_execution_artifact::ProgramExecutionRequest;
use serde::Serialize as _;

#[wasm_bindgen(js_name = executeProgramArtifact)]
pub async fn execute_program_artifact(request: JsValue) -> Result<JsValue, JsValue> {
    runmat_runtime::builtins::wasm_registry::register_all();
    crate::api::init::ensure_internal_builtins();
    let request =
        serde_wasm_bindgen::from_value::<ProgramExecutionRequest>(request).map_err(|error| {
            JsValue::from_str(&format!("invalid program execution request: {error}"))
        })?;
    let response = runmat_vm::execute_program_request(request).await;
    let serializer =
        serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true);
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("program response encoding failed: {error}")))
}
