use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::Future;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;

use runmat_process_host::ipc::{
    authenticate_host_blocking, read_payload_blocking, write_payload_blocking, FrameLimits,
    HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_types::{CallableIdentity, SymbolName};
use runmat_value::Value;

use super::{
    decode_portable, encode_portable, wire_error, PythonCallbackRequest, PythonDriverMessage,
    PythonHostMessage, PythonInvocationRequest, PythonInvocationResult, PythonRemoteReference,
    PythonShutdownResult, PythonWireError, PythonWireValue, PYTHON_HOST_CONFIG_ENV,
    PYTHON_HOST_MAX_CALLBACK_DEPTH, PYTHON_HOST_MAX_MESSAGE_BYTES, PYTHON_HOST_PROTOCOL,
    PYTHON_HOST_SCHEMA_VERSION, PYTHON_HOST_SECRET_ENV, PYTHON_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{
    ForeignCall, RuntimeCallRequest, RuntimeCallService, RuntimeContext, RuntimeServicePorts,
};
use crate::execution::RuntimeExecutionService;
use crate::foreign::{
    ForeignAdapter, ForeignPlatform, ForeignRuntime, PythonAdapter, PythonRuntimeConfiguration,
};
use crate::RuntimeError;

const CALLBACK_PREFIX: &str = "__runmat_python_isolated_callback_";
type StdioChannel = HostChannel<std::io::Stdin, std::io::Stdout>;

pub fn run_python_extension_host() -> Result<(), String> {
    let encoded_secret = std::env::var(PYTHON_HOST_SECRET_ENV)
        .map_err(|_| "Python host bootstrap secret is unavailable".to_string())?;
    std::env::remove_var(PYTHON_HOST_SECRET_ENV);
    let secret = SessionSecret::from_hex(&encoded_secret).map_err(|error| error.to_string())?;
    drop(encoded_secret);
    let encoded_config = std::env::var(PYTHON_HOST_CONFIG_ENV)
        .map_err(|_| "Python host configuration is unavailable".to_string())?;
    std::env::remove_var(PYTHON_HOST_CONFIG_ENV);
    let mut configuration: PythonRuntimeConfiguration =
        serde_json::from_str(&encoded_config).map_err(|error| error.to_string())?;
    configuration.execution_mode = runmat_python::PythonExecutionMode::InProcess;
    let snapshot_root = std::env::var_os(PYTHON_HOST_SNAPSHOT_ROOT_ENV)
        .ok_or_else(|| "Python host snapshot root is unavailable".to_string())?;
    std::env::remove_var(PYTHON_HOST_SNAPSHOT_ROOT_ENV);
    let snapshots = Rc::new(
        SharedSnapshotStore::open_existing(PathBuf::from(snapshot_root))
            .map_err(|error| error.to_string())?,
    );
    let mut reader = std::io::stdin();
    let mut writer = std::io::stdout();
    let session = authenticate_host_blocking(
        &mut reader,
        &mut writer,
        HostHandshake::new(
            PYTHON_HOST_PROTOCOL,
            PYTHON_HOST_SCHEMA_VERSION,
            PYTHON_HOST_MAX_MESSAGE_BYTES,
        ),
        &secret,
    )
    .map_err(|error| error.to_string())?;
    let channel = Rc::new(RefCell::new(HostChannel {
        reader,
        writer,
        limits: session.limits,
    }));
    let resources = Rc::new(RefCell::new(RemoteResources::default()));
    let callback_router = Rc::new(HostCallbackRouter {
        channel: Rc::clone(&channel),
        snapshots: Rc::clone(&snapshots),
        resources: Rc::clone(&resources),
        active_request: Cell::new(0),
        depth: Cell::new(0),
    });
    let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
    let adapter = PythonAdapter::with_configuration(foreign.handles().clone(), configuration)
        .map_err(|error| error.to_string())?;
    foreign
        .register_adapter(adapter.clone())
        .map_err(|error| error.to_string())?;
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_foreign(foreign)
            .with_call(callback_router.clone()),
    );
    ACTIVE_HOST.with(|host| {
        host.replace(Some((adapter.clone(), context.clone())));
    });

    let outcome = loop {
        let message = channel.borrow_mut().read_driver()?;
        match message {
            PythonDriverMessage::Invoke(request) => {
                let request_id = request.request_id;
                callback_router.active_request.set(request_id);
                let outcome = invoke_request(&adapter, &context, request, &snapshots, &resources);
                callback_router.active_request.set(0);
                channel
                    .borrow_mut()
                    .write_host(&PythonHostMessage::Invocation(PythonInvocationResult {
                        request_id,
                        outcome,
                    }))?;
            }
            PythonDriverMessage::Shutdown {
                request_id,
                releases,
            } => {
                resources.borrow_mut().release(&releases);
                channel
                    .borrow_mut()
                    .write_host(&PythonHostMessage::Shutdown(PythonShutdownResult {
                        request_id,
                        outcome: Ok(()),
                    }))?;
                break Ok(());
            }
            PythonDriverMessage::CallbackResult(_) => {
                break Err("callback result arrived outside an active callback".into());
            }
        }
    };
    ACTIVE_HOST.with(|host| {
        host.replace(None);
    });
    outcome
}

fn invoke_request(
    adapter: &Rc<PythonAdapter>,
    context: &RuntimeContext,
    request: PythonInvocationRequest,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<PythonWireValue, PythonWireError> {
    resources.borrow_mut().release(&request.releases);
    let arguments = request
        .arguments
        .into_iter()
        .map(|value| decode_host_value(value, snapshots, resources))
        .collect::<Result<Vec<_>, _>>()?;
    let value = pollster::block_on(context.scope(adapter.invoke(
        context.clone(),
        ForeignCall {
            adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
            symbol: request.operation,
            arguments,
            requested_outputs: request.requested_outputs as usize,
        },
    )))
    .map_err(runtime_wire_error)?;
    encode_host_value(value, snapshots, resources)
}

fn decode_host_value(
    value: PythonWireValue,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<Value, PythonWireError> {
    match value {
        PythonWireValue::Portable(transfer) => decode_portable(&transfer, snapshots),
        PythonWireValue::Foreign(reference) => resources
            .borrow()
            .values
            .get(&reference.id)
            .cloned()
            .ok_or_else(|| wire_error("RunMat:Python:StaleHandle", "Python object was released")),
        PythonWireValue::Callback { id } => {
            Ok(Value::FunctionHandle(format!("{CALLBACK_PREFIX}{id}")))
        }
        PythonWireValue::OutputList(values) => values
            .into_iter()
            .map(|value| decode_host_value(value, snapshots, resources))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::OutputList),
    }
}

fn encode_host_value(
    value: Value,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<PythonWireValue, PythonWireError> {
    match value {
        Value::Foreign(reference) => {
            let mut resources = resources.borrow_mut();
            let identity = (
                reference.host_identity.clone(),
                reference.handle,
                reference.generation,
            );
            let id = if let Some(id) = resources.identities.get(&identity) {
                *id
            } else {
                let id = resources.next_id;
                resources.next_id = resources.next_id.checked_add(1).ok_or_else(|| {
                    wire_error("RunMat:Python:HostProtocol", "remote identity exhausted")
                })?;
                resources.identities.insert(identity, id);
                resources
                    .values
                    .insert(id, Value::Foreign(reference.clone()));
                id
            };
            Ok(PythonWireValue::Foreign(PythonRemoteReference {
                id,
                type_identity: reference.type_identity,
                ownership: reference.ownership,
                affinity: reference.affinity,
                lifetime: reference.lifetime,
            }))
        }
        Value::OutputList(values) => values
            .into_iter()
            .map(|value| encode_host_value(value, snapshots, resources))
            .collect::<Result<Vec<_>, _>>()
            .map(PythonWireValue::OutputList),
        other => encode_portable(&other, snapshots).map(PythonWireValue::Portable),
    }
}

struct HostCallbackRouter {
    channel: Rc<RefCell<StdioChannel>>,
    snapshots: Rc<SharedSnapshotStore>,
    resources: Rc<RefCell<RemoteResources>>,
    active_request: Cell<u64>,
    depth: Cell<u16>,
}

impl RuntimeCallService for HostCallbackRouter {
    fn resolve(&self, _name: &str) -> Option<usize> {
        None
    }

    fn invoke(
        &self,
        request: RuntimeCallRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value, RuntimeError>> + 'static>> {
        let result = self
            .invoke_now(request)
            .map_err(super::client::runtime_error);
        Box::pin(async move { result })
    }
}

impl HostCallbackRouter {
    fn invoke_now(&self, request: RuntimeCallRequest) -> Result<Value, PythonWireError> {
        let callback_id = callback_id(&request.identity).ok_or_else(|| {
            wire_error(
                "RunMat:Python:CallbackProtocol",
                "isolated Python host received a non-driver callback identity",
            )
        })?;
        let request_id = self.active_request.get();
        if request_id == 0 {
            return Err(wire_error(
                "RunMat:Python:CallbackProtocol",
                "Python callback has no active invocation",
            ));
        }
        let depth = self.depth.get().saturating_add(1);
        if depth > PYTHON_HOST_MAX_CALLBACK_DEPTH {
            return Err(wire_error(
                "RunMat:Python:CallbackProtocol",
                "Python callback depth limit exceeded",
            ));
        }
        self.depth.set(depth);
        let result = (|| {
            let arguments = request
                .arguments
                .into_iter()
                .map(|value| encode_host_value(value, &self.snapshots, &self.resources))
                .collect::<Result<Vec<_>, _>>()?;
            self.channel
                .borrow_mut()
                .write_host(&PythonHostMessage::Callback(PythonCallbackRequest {
                    request_id,
                    callback_id,
                    depth,
                    arguments,
                }))
                .map_err(|error| wire_error("RunMat:Python:HostTransport", error))?;
            loop {
                let message = self
                    .channel
                    .borrow_mut()
                    .read_driver()
                    .map_err(|error| wire_error("RunMat:Python:HostTransport", error))?;
                match message {
                    PythonDriverMessage::CallbackResult(result)
                        if result.request_id == request_id && result.callback_id == callback_id =>
                    {
                        let value = result.outcome?;
                        return decode_host_value(value, &self.snapshots, &self.resources);
                    }
                    PythonDriverMessage::Invoke(nested) => {
                        let nested_id = nested.request_id;
                        let adapter = current_python_adapter()?;
                        let context = current_python_context()?;
                        let outcome = invoke_request(
                            &adapter,
                            &context,
                            nested,
                            &self.snapshots,
                            &self.resources,
                        );
                        self.channel
                            .borrow_mut()
                            .write_host(&PythonHostMessage::Invocation(PythonInvocationResult {
                                request_id: nested_id,
                                outcome,
                            }))
                            .map_err(|error| wire_error("RunMat:Python:HostTransport", error))?;
                    }
                    _ => {
                        return Err(wire_error(
                            "RunMat:Python:CallbackProtocol",
                            "unexpected driver record while Python callback was active",
                        ));
                    }
                }
            }
        })();
        self.depth.set(depth - 1);
        result
    }
}

thread_local! {
    static ACTIVE_HOST: RefCell<Option<(Rc<PythonAdapter>, RuntimeContext)>> = const { RefCell::new(None) };
}

fn current_python_adapter() -> Result<Rc<PythonAdapter>, PythonWireError> {
    ACTIVE_HOST
        .with(|host| host.borrow().as_ref().map(|value| value.0.clone()))
        .ok_or_else(|| {
            wire_error(
                "RunMat:Python:HostProtocol",
                "Python host adapter is unavailable",
            )
        })
}

fn current_python_context() -> Result<RuntimeContext, PythonWireError> {
    ACTIVE_HOST
        .with(|host| host.borrow().as_ref().map(|value| value.1.clone()))
        .ok_or_else(|| {
            wire_error(
                "RunMat:Python:HostProtocol",
                "Python host context is unavailable",
            )
        })
}

fn callback_id(identity: &CallableIdentity) -> Option<u64> {
    let CallableIdentity::DynamicName(SymbolName(name)) = identity else {
        return None;
    };
    name.strip_prefix(CALLBACK_PREFIX)?.parse().ok()
}

struct RemoteResources {
    next_id: u64,
    values: BTreeMap<u64, Value>,
    identities: BTreeMap<(String, u64, u64), u64>,
}

impl Default for RemoteResources {
    fn default() -> Self {
        Self {
            next_id: 1,
            values: BTreeMap::new(),
            identities: BTreeMap::new(),
        }
    }
}

impl RemoteResources {
    fn release(&mut self, ids: &[u64]) {
        for id in ids {
            let Some(Value::Foreign(reference)) = self.values.remove(id) else {
                continue;
            };
            self.identities.remove(&(
                reference.host_identity,
                reference.handle,
                reference.generation,
            ));
        }
    }
}

struct HostChannel<R, W> {
    reader: R,
    writer: W,
    limits: FrameLimits,
}

impl<R: Read, W: Write> HostChannel<R, W> {
    fn read_driver(&mut self) -> Result<PythonDriverMessage, String> {
        let payload = read_payload_blocking(&mut self.reader, self.limits)
            .map_err(|error| error.to_string())?;
        let message: PythonDriverMessage = serde_json::from_slice(&payload)
            .map_err(|error| format!("invalid Python driver record: {error}"))?;
        message
            .validate()
            .map_err(|error| format!("invalid Python driver record: {error}"))?;
        Ok(message)
    }

    fn write_host(&mut self, message: &PythonHostMessage) -> Result<(), String> {
        message
            .validate()
            .map_err(|error| format!("invalid Python host record: {error}"))?;
        let payload = serde_json::to_vec(message)
            .map_err(|error| format!("could not encode Python host record: {error}"))?;
        write_payload_blocking(&mut self.writer, &payload, self.limits)
            .map_err(|error| error.to_string())
    }
}

fn runtime_wire_error(error: RuntimeError) -> PythonWireError {
    let python = error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<runmat_python::PythonError>())
        .cloned();
    PythonWireError {
        identifier: error
            .identifier()
            .unwrap_or("RunMat:Python:RuntimeError")
            .into(),
        message: error.to_string(),
        python,
    }
}
