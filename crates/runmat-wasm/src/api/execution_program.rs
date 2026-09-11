use wasm_bindgen::prelude::*;

use runmat_execution_artifact::ProgramExecutionRequest;
use serde::Serialize as _;

#[derive(serde::Deserialize)]
struct ProgramRequestSchemaHeader {
    schema_version: u16,
}

#[wasm_bindgen(js_name = executeProgramArtifact)]
pub async fn execute_program_artifact(request: JsValue) -> Result<JsValue, JsValue> {
    runmat_runtime::builtins::wasm_registry::register_all();
    crate::api::init::ensure_internal_builtins();
    let header = serde_wasm_bindgen::from_value::<ProgramRequestSchemaHeader>(request.clone())
        .map_err(|error| {
            JsValue::from_str(&format!("invalid program request envelope: {error}"))
        })?;
    if header.schema_version != runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
    {
        return Err(JsValue::from_str(&format!(
            "unsupported program execution request schema {}; expected {}; rebuild the request with the current value-payload codec",
            header.schema_version,
            runmat_execution_artifact::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION
        )));
    }
    let request =
        serde_wasm_bindgen::from_value::<ProgramExecutionRequest>(request).map_err(|error| {
            JsValue::from_str(&format!("invalid program execution request: {error}"))
        })?;
    let response = crate::runtime::execution::execute_local_program(request).await;
    let serializer =
        serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true);
    response
        .serialize(&serializer)
        .map_err(|error| JsValue::from_str(&format!("program response encoding failed: {error}")))
}
