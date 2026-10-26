use js_sys::{Function, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use runmat_execution_artifact::ProgramExecutionResponse;

use super::request::BrowserWorkerRequest;
use serde::Serialize;

use super::model::BrowserExecutionCapabilities;

mod owned_response;
mod program;

const BROWSER_EXECUTION_SUBMISSION_SCHEMA_V2: u16 = 2;

#[derive(Clone)]
pub(crate) struct BrowserExecutionHost {
    target: JsValue,
    capabilities: BrowserExecutionCapabilities,
    worker_inventory: Option<std::rc::Rc<runmat_execution_artifact::ProgramWorkerInventory>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedPhysicalExit {
    Terminated,
    NotRunning,
}

impl BrowserExecutionHost {
    pub(crate) fn new(target: JsValue) -> Result<Self, JsValue> {
        let capabilities_value = Reflect::get(&target, &JsValue::from_str("capabilities"))?;
        let capabilities =
            serde_wasm_bindgen::from_value::<BrowserExecutionCapabilities>(capabilities_value)
                .map_err(|error| {
                    JsValue::from_str(&format!("invalid executionHost capabilities: {error}"))
                })?
                .validate()
                .map_err(|error| JsValue::from_str(&error))?;
        required_method(&target, "launch")?;
        required_method(&target, "cancel")?;
        if capabilities.owned_builtin_work {
            required_method(&target, "cancelOwned")?;
        }
        if capabilities.selected_program_work {
            required_method(&target, "launchProgram")?;
            required_method(&target, "cancelProgram")?;
        }
        let worker_inventory = Reflect::get(&target, &JsValue::from_str("workerInventory"))?;
        let worker_inventory = if worker_inventory.is_undefined() || worker_inventory.is_null() {
            if capabilities.selected_program_work {
                return Err(JsValue::from_str(
                    "selectedProgramWork requires its actual workerInventory",
                ));
            }
            None
        } else {
            let inventory: runmat_execution_artifact::ProgramWorkerInventory =
                serde_wasm_bindgen::from_value(worker_inventory).map_err(|error| {
                    JsValue::from_str(&format!("invalid worker inventory: {error}"))
                })?;
            inventory
                .validate()
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
            if inventory.host.target != runmat_execution::host::ExecutionHostTarget::BrowserWasm {
                return Err(JsValue::from_str(
                    "browser program host requires a BrowserWasm worker inventory",
                ));
            }
            Some(std::rc::Rc::new(inventory))
        };
        Ok(Self {
            target,
            capabilities,
            worker_inventory,
        })
    }

    pub(crate) fn capabilities(&self) -> BrowserExecutionCapabilities {
        self.capabilities
    }

    pub(crate) fn worker_inventory(
        &self,
    ) -> Option<&runmat_execution_artifact::ProgramWorkerInventory> {
        self.worker_inventory.as_deref()
    }

    pub(crate) async fn launch(
        &self,
        task_id: &str,
        worker_id: &str,
        request: &BrowserWorkerRequest,
    ) -> Result<ProgramExecutionResponse, String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct LaunchRequest<'a> {
            schema_version: u16,
            task_id: &'a str,
            worker_id: &'a str,
            program: &'a BrowserWorkerRequest,
        }
        let owned_request = match request {
            BrowserWorkerRequest::OwnedBuiltin(request) => Some(request),
            _ => None,
        };
        let launch_error = |error: JsValue| {
            if owned_request.is_some() {
                "owned worker launch failed before a verified response".to_string()
            } else {
                js_error(error)
            }
        };
        let function = required_method(&self.target, "launch").map_err(&launch_error)?;
        let serializer =
            serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true);
        let payload = LaunchRequest {
            schema_version: BROWSER_EXECUTION_SUBMISSION_SCHEMA_V2,
            task_id,
            worker_id,
            program: request,
        }
        .serialize(&serializer)
        .map_err(|error| {
            if owned_request.is_some() {
                "owned worker request could not be encoded".to_string()
            } else {
                error.to_string()
            }
        })?;
        let result = function
            .call1(&self.target, &payload)
            .map_err(&launch_error)?;
        let resolved = JsFuture::from(Promise::resolve(&result))
            .await
            .map_err(&launch_error)?;
        if let Some(request) = owned_request {
            owned_response::preflight(&resolved, request)?;
        }
        serde_wasm_bindgen::from_value(resolved).map_err(|error| {
            if owned_request.is_some() {
                "owned worker response could not be decoded after preflight".to_string()
            } else {
                error.to_string()
            }
        })
    }

    pub(crate) fn cancel(&self, task_id: &str) -> Result<(), String> {
        let function = required_method(&self.target, "cancel").map_err(js_error)?;
        function
            .call1(&self.target, &JsValue::from_str(task_id))
            .map_err(js_error)?;
        Ok(())
    }

    /// A fulfilled Promise alone is not evidence of physical worker exit.
    /// The trusted host must return a task-bound termination or not-running
    /// acknowledgment. Rejection and malformed acknowledgments prove nothing.
    pub(crate) async fn cancel_owned(&self, task_id: &str) -> Result<OwnedPhysicalExit, String> {
        let function = required_method(&self.target, "cancelOwned")
            .map_err(|_| "owned worker cancellation method is unavailable".to_string())?;
        let value = function
            .call1(&self.target, &JsValue::from_str(task_id))
            .map_err(|_| "owned worker cancellation invocation failed".to_string())?;
        let promise = value.dyn_into::<Promise>().map_err(|_| {
            "executionHost.cancelOwned must return a task-bound physical-exit Promise".to_string()
        })?;
        let acknowledgment = JsFuture::from(promise)
            .await
            .map_err(|_| "owned worker cancellation was not acknowledged".to_string())?;
        owned_response::preflight_exit_ack(&acknowledgment, task_id)?;
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct ExitAcknowledgment {
            task_id: String,
            state: ExitState,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        enum ExitState {
            Terminated,
            NotRunning,
        }
        let acknowledgment: ExitAcknowledgment = serde_wasm_bindgen::from_value(acknowledgment)
            .map_err(|_| "owned worker cancellation acknowledgment is invalid".to_string())?;
        if acknowledgment.task_id != task_id {
            return Err("executionHost.cancelOwned acknowledged a different task".into());
        }
        Ok(match acknowledgment.state {
            ExitState::Terminated => OwnedPhysicalExit::Terminated,
            ExitState::NotRunning => OwnedPhysicalExit::NotRunning,
        })
    }
}

pub(crate) fn execution_host_from_options(
    options: &JsValue,
) -> Result<Option<BrowserExecutionHost>, JsValue> {
    if options.is_null() || options.is_undefined() || !options.is_object() {
        return Ok(None);
    }
    let value = Reflect::get(options, &JsValue::from_str("executionHost"))?;
    if value.is_null() || value.is_undefined() {
        Ok(None)
    } else {
        BrowserExecutionHost::new(value).map(Some)
    }
}

fn required_method(target: &JsValue, name: &str) -> Result<Function, JsValue> {
    Reflect::get(target, &JsValue::from_str(name))?
        .dyn_into::<Function>()
        .map_err(|_| JsValue::from_str(&format!("executionHost.{name} must be a function")))
}

fn js_error(value: JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| format!("browser execution host rejected: {value:?}"))
}
