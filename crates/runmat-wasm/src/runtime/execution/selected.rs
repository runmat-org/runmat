//! Assignment-bound physical worker invocation. The Runtime owns cancellation,
//! workspace cleanup, and the terminal retirement receipt; JavaScript transports it.
use super::{BrowserExecutionService, BrowserWorkerRequest};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{CancellationReason, ProgramExecutionAssignment};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::context::{BrowserLocalTaskSupervisor, RuntimeContext, RuntimeLifecycleHost};
use runmat_runtime::workspace::{BaseWorkspaceOwner, BaseWorkspaceService};
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProgramWorkerInventory {
    pub schema_version: u16,
    pub host: runmat_execution::host::ExecutionHostInventory,
    pub accelerators: Vec<runmat_execution::resource::AcceleratorDevice>,
}

pub(crate) fn inventory() -> Result<ProgramWorkerInventory, String> {
    let accelerators = super::resources::browser_accelerators().map_err(|e| e.to_string())?;
    Ok(ProgramWorkerInventory {
        schema_version: 1,
        host: super::resources::browser_execution_host_inventory(&accelerators)
            .map_err(|e| e.to_string())?,
        accelerators,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProgramTaskCancellation {
    pub schema_version: u16,
    pub assignment: ProgramExecutionAssignment,
    pub reason: CancellationReason,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ProgramRetirement {
    Retired,
    Unverified { diagnostic: String },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProgramTaskTerminal {
    pub schema_version: u16,
    pub assignment: ProgramExecutionAssignment,
    pub response: ProgramExecutionResponse,
    pub retirement: ProgramRetirement,
}
impl ProgramTaskTerminal {
    pub(crate) fn verify(&self, assignment: &ProgramExecutionAssignment) -> Result<(), String> {
        if self.schema_version != 1 || &self.assignment != assignment {
            return Err("selected worker terminal belongs to another invocation".into());
        }
        match &self.retirement {
            ProgramRetirement::Retired => Ok(()),
            ProgramRetirement::Unverified { diagnostic } => Err(format!(
                "selected worker retirement is unverified: {diagnostic}"
            )),
        }
    }
}

pub(crate) fn assignment(request: &BrowserWorkerRequest) -> Option<&ProgramExecutionAssignment> {
    match request {
        BrowserWorkerRequest::Direct(r) => r.assignment.as_ref(),
        BrowserWorkerRequest::Graph(r) => r.assignment.as_ref(),
        BrowserWorkerRequest::OwnedBuiltin(r) => r.assignment.as_ref(),
    }
}

#[wasm_bindgen]
pub struct WorkerProgramInvocation {
    assignment: ProgramExecutionAssignment,
    request: RefCell<Option<BrowserWorkerRequest>>,
    cancellation: CancellationSource,
    started: Cell<bool>,
}

impl WorkerProgramInvocation {
    pub(crate) fn new(request: BrowserWorkerRequest) -> Result<Self, String> {
        request.validate()?;
        match &request {
            BrowserWorkerRequest::Direct(r) => super::admission::validate_browser_admission(r)?,
            BrowserWorkerRequest::Graph(r) => {
                super::admission::validate_browser_graph_admission(r)?
            }
            BrowserWorkerRequest::OwnedBuiltin(_) => {
                return Err("owned builtin work uses its existing owned invocation protocol".into())
            }
        }
        let assignment = assignment(&request)
            .ok_or("selected worker request has no scheduler assignment")?
            .clone();
        assignment.validate().map_err(|e| e.to_string())?;
        if assignment.backend != runmat_execution::PoolBackend::BrowserWorkers {
            return Err("selected worker request is not assigned to browser workers".into());
        }
        // Stock workers currently provision CPU execution only. Never borrow
        // a UI realm's accelerator identity or claim its device lease.
        if !assignment.resources.accelerator_leases.is_empty() {
            return Err("selected CPU worker cannot execute an accelerator lease".into());
        }
        Ok(Self {
            assignment,
            request: RefCell::new(Some(request)),
            cancellation: CancellationSource::new(),
            started: Cell::new(false),
        })
    }
}

#[wasm_bindgen]
impl WorkerProgramInvocation {
    #[wasm_bindgen(getter)]
    pub fn assignment(&self) -> Result<JsValue, JsValue> {
        encode(&self.assignment)
    }

    pub fn cancel(&self, request: JsValue) -> Result<(), JsValue> {
        let request: ProgramTaskCancellation =
            serde_wasm_bindgen::from_value(request).map_err(js_error)?;
        if request.schema_version != 1 || request.assignment != self.assignment {
            return Err(JsValue::from_str(
                "selected cancellation belongs to another invocation",
            ));
        }
        self.cancellation.cancel(request.reason);
        Ok(())
    }

    pub async fn run(&self) -> Result<JsValue, JsValue> {
        if self.started.replace(true) {
            return Err(JsValue::from_str("selected invocation has already run"));
        }
        let request = self
            .request
            .borrow_mut()
            .take()
            .ok_or_else(|| JsValue::from_str("selected invocation has no request"))?;
        let service = BrowserExecutionService::new(None).map_err(js_error)?;
        let runtime = RuntimeContext::with_cancellation(service, self.cancellation.token());
        let owner = BaseWorkspaceOwner::new();
        let workspace = Rc::new(BaseWorkspaceService::new(owner.clone()));
        let supervisor = Rc::new(BrowserLocalTaskSupervisor::new());
        let ports = runtime
            .service_ports()
            .clone()
            .with_workspace(workspace.clone())
            .with_base_workspace(workspace)
            .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
        let runtime = runtime.with_service_ports(ports);
        let host = RuntimeLifecycleHost::new(&runtime, supervisor).map_err(js_error)?;
        let response = match request {
            BrowserWorkerRequest::Direct(request) => {
                runmat_vm::execute_program_request_with_context(request, runtime.clone()).await
            }
            BrowserWorkerRequest::Graph(request) => {
                runmat_vm::execute_graph_request_with_context(request, runtime.clone()).await
            }
            BrowserWorkerRequest::OwnedBuiltin(_) => unreachable!(),
        };
        // Preserve the original response even when cleanup fails. Cleanup is
        // separate evidence, never a replacement for the program's diagnostic.
        let drained = host.retire_workspace_and_drain(&runtime).await;
        host.force_close();
        let closed = owner.close();
        let retirement = match (drained, closed) {
            (Ok(()), Ok(_)) => ProgramRetirement::Retired,
            (drain, close) => ProgramRetirement::Unverified {
                diagnostic: format!(
                    "{}{}",
                    drain
                        .err()
                        .map(|e| format!("Runtime cleanup: {e}"))
                        .unwrap_or_default(),
                    close
                        .err()
                        .map(|e| format!(" Workspace close: {e}"))
                        .unwrap_or_default()
                ),
            },
        };
        drop(host);
        drop(runtime);
        encode(&ProgramTaskTerminal {
            schema_version: 1,
            assignment: self.assignment.clone(),
            response,
            retirement,
        })
    }
}

pub(crate) fn encode(value: &impl Serialize) -> Result<JsValue, JsValue> {
    value
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .map_err(js_error)
}
fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
