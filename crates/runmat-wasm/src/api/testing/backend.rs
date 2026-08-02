use std::fmt;

use js_sys::{Function, Promise, Reflect};
use runmat_test::protocol::{ProtocolHandshake, WorkerCapability};
use runmat_test_runner::host::{HostCapabilities, IsolationMode};
use runmat_test_runner::worker::{
    BackendCapabilities, BackendError, BackendErrorKind, BackendFuture, CancelRequest,
    ExecutionRequest, SpawnRequest, WorkerBackend, WorkerExecution,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

#[derive(Clone)]
pub(super) struct JsWorkerBackend {
    target: JsValue,
    capabilities: BackendCapabilities,
}

impl JsWorkerBackend {
    pub fn new(target: JsValue) -> Result<Self, BackendError> {
        let value = sync_call(&target, "capabilities", Vec::new())?;
        let wire: CapabilityWire = serde_wasm_bindgen::from_value(value).map_err(protocol_error)?;
        let isolation = wire
            .isolation
            .into_iter()
            .map(|value| match value {
                IsolationWire::Worker => IsolationMode::Worker,
                IsolationWire::Session => IsolationMode::Session,
                IsolationWire::None => IsolationMode::None,
            })
            .collect::<Vec<_>>();
        let host = HostCapabilities::new(isolation, wire.max_workers)
            .map_err(|error| BackendError::new(BackendErrorKind::Rejected, error.to_string()))?;
        Ok(Self {
            target,
            capabilities: BackendCapabilities {
                host,
                handshake: ProtocolHandshake::current(
                    "runmat-browser-coordinator",
                    vec![
                        WorkerCapability::SessionIsolation,
                        WorkerCapability::CapturedOutput,
                        WorkerCapability::Artifacts,
                    ],
                ),
            },
        })
    }
}

#[derive(Clone)]
pub(super) struct JsWorkerSession {
    id: String,
    handle: JsValue,
}

impl fmt::Debug for JsWorkerSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JsWorkerSession")
            .field("id", &self.id)
            .finish()
    }
}

impl PartialEq for JsWorkerSession {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for JsWorkerSession {}

impl WorkerBackend for JsWorkerBackend {
    type Session = JsWorkerSession;

    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }

    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session> {
        Box::pin(async move {
            let input = SpawnWire {
                plan: request.submission.plan,
                snapshot: request.submission.snapshot,
                isolation: request.isolation,
            };
            let output = async_call(&self.target, "spawn", vec![encode(&input)?]).await?;
            let id = Reflect::get(&output, &JsValue::from_str("id"))
                .map_err(js_backend_error)?
                .as_string()
                .ok_or_else(|| {
                    BackendError::new(
                        BackendErrorKind::MalformedProtocol,
                        "browser worker session id must be a string",
                    )
                })?;
            Ok(JsWorkerSession { id, handle: output })
        })
    }

    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        Box::pin(async move {
            let output = async_call(
                &self.target,
                "execute",
                vec![session.handle.clone(), encode(&request_wire(request))?],
            )
            .await?;
            serde_wasm_bindgen::from_value(output).map_err(protocol_error)
        })
    }

    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>> {
        Box::pin(async move {
            let output = async_call(
                &self.target,
                "cancel",
                vec![session.handle.clone(), encode(&cancel_wire(request))?],
            )
            .await?;
            if output.is_null() || output.is_undefined() {
                Ok(None)
            } else {
                serde_wasm_bindgen::from_value(output)
                    .map(Some)
                    .map_err(protocol_error)
            }
        })
    }

    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            async_call(&self.target, "terminate", vec![session.handle.clone()])
                .await
                .map(|_| ())
        })
    }

    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()> {
        Box::pin(async move {
            async_call(&self.target, "shutdown", vec![session.handle.clone()])
                .await
                .map(|_| ())
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CapabilityWire {
    isolation: Vec<IsolationWire>,
    max_workers: usize,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum IsolationWire {
    Worker,
    Session,
    None,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SpawnWire {
    plan: runmat_test::plan::TestPlan,
    snapshot: runmat_test::discovery::FrozenTestRunSnapshot,
    isolation: IsolationMode,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExecutionWire {
    test_id: runmat_test::identity::TestId,
    attempt: u32,
    deadline_ms: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CancelWire {
    run_id: runmat_test::identity::RunId,
    reason: String,
    grace_deadline_ms: u64,
}

fn request_wire(request: ExecutionRequest) -> ExecutionWire {
    ExecutionWire {
        test_id: request.test_id,
        attempt: request.attempt,
        deadline_ms: request.deadline_ms,
    }
}

fn cancel_wire(request: CancelRequest) -> CancelWire {
    CancelWire {
        run_id: request.run_id,
        reason: request.reason,
        grace_deadline_ms: request.grace_deadline_ms,
    }
}

fn encode(value: &impl Serialize) -> Result<JsValue, BackendError> {
    serde_wasm_bindgen::to_value(value).map_err(protocol_error)
}

fn sync_call(
    target: &JsValue,
    name: &str,
    arguments: Vec<JsValue>,
) -> Result<JsValue, BackendError> {
    let function = Reflect::get(target, &JsValue::from_str(name))
        .map_err(js_backend_error)?
        .dyn_into::<Function>()
        .map_err(|_| {
            BackendError::new(
                BackendErrorKind::Rejected,
                format!("{name} is not a function"),
            )
        })?;
    let array = js_sys::Array::new();
    for argument in arguments {
        array.push(&argument);
    }
    function.apply(target, &array).map_err(js_backend_error)
}

async fn async_call(
    target: &JsValue,
    name: &str,
    arguments: Vec<JsValue>,
) -> Result<JsValue, BackendError> {
    let value = sync_call(target, name, arguments)?;
    JsFuture::from(Promise::resolve(&value))
        .await
        .map_err(js_backend_error)
}

fn protocol_error(error: impl fmt::Display) -> BackendError {
    BackendError::new(BackendErrorKind::MalformedProtocol, error.to_string())
}

fn js_backend_error(value: JsValue) -> BackendError {
    BackendError::new(
        BackendErrorKind::Transport,
        value
            .as_string()
            .unwrap_or_else(|| "browser worker adapter rejected the operation".into()),
    )
}
