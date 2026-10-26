//! Per-invocation cancellation, independent of the interactive UI token.

use std::cell::RefCell;

use runmat_execution::{CancellationSource, ProgramExecutionAssignment};
use runmat_execution_artifact::ProgramTaskCancellation;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::runtime::execution::{controlled, BrowserWorkerRequest};

#[wasm_bindgen]
pub struct WorkerProgramInvocation {
    request: RefCell<Option<BrowserWorkerRequest>>,
    assignment: ProgramExecutionAssignment,
    cancellation: CancellationSource,
}

/// Describe the exact stock worker factory from this loaded module. It does
/// not create a probe worker, session, or a second execution runtime.
#[wasm_bindgen(js_name = workerProgramInventory)]
pub fn worker_program_inventory() -> Result<JsValue, JsValue> {
    runmat_runtime::builtins::wasm_registry::register_all();
    encode(&controlled::inventory().map_err(js_error)?)
}

#[wasm_bindgen(js_name = createWorkerProgramInvocation)]
pub fn create_worker_program_invocation(
    request: JsValue,
) -> Result<WorkerProgramInvocation, JsValue> {
    runmat_runtime::builtins::wasm_registry::register_all();
    let request = serde_wasm_bindgen::from_value::<BrowserWorkerRequest>(request)
        .map_err(|error| js_error(format!("invalid selected worker request: {error}")))?;
    request.validate().map_err(js_error)?;
    let assignment = controlled::assignment(&request).map_err(js_error)?.clone();
    Ok(WorkerProgramInvocation {
        request: RefCell::new(Some(request)),
        assignment,
        cancellation: CancellationSource::new(),
    })
}

#[wasm_bindgen]
impl WorkerProgramInvocation {
    #[wasm_bindgen(getter)]
    pub fn assignment(&self) -> Result<JsValue, JsValue> {
        encode(&self.assignment)
    }

    /// A cancellation return is only request delivery. The host must await
    /// run's terminal retirement proof before releasing the selected grant.
    pub fn cancel(&self, request: JsValue) -> Result<(), JsValue> {
        let request = serde_wasm_bindgen::from_value::<ProgramTaskCancellation>(request)
            .map_err(|error| js_error(format!("invalid program cancellation: {error}")))?;
        request
            .validate_assignment(&self.assignment)
            .map_err(|error| js_error(error.to_string()))?;
        self.cancellation.cancel(request.reason);
        Ok(())
    }

    pub async fn run(&self) -> Result<JsValue, JsValue> {
        let request = self
            .request
            .borrow_mut()
            .take()
            .ok_or_else(|| js_error("selected worker invocation was already started"))?;
        let terminal =
            controlled::execute(request, self.assignment.clone(), self.cancellation.token()).await;
        encode(&terminal)
    }
}

fn encode(value: &impl Serialize) -> Result<JsValue, JsValue> {
    value
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .map_err(|error| js_error(format!("program task protocol encoding failed: {error}")))
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
