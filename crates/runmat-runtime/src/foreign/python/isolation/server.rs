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
    decode_portable, encode_portable, root_plain_value, wire_error, PythonCallbackRequest,
    PythonDriverMessage, PythonHostMessage, PythonInvocationRequest, PythonInvocationResult,
    PythonRemoteReference, PythonShutdownResult, PythonWireError, PythonWireValue,
    PYTHON_HOST_CONFIG_ENV, PYTHON_HOST_MAX_CALLBACK_DEPTH, PYTHON_HOST_MAX_MESSAGE_BYTES,
    PYTHON_HOST_PROTOCOL, PYTHON_HOST_SCHEMA_VERSION, PYTHON_HOST_SECRET_ENV,
    PYTHON_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{
    ForeignCall, RuntimeCallRequest, RuntimeCallService, RuntimeContext, RuntimeLifecycleHost,
    RuntimeLocalTaskSupervisor, RuntimeServicePorts,
};
use crate::execution::{RootedValueMember, RootedValuesBuilder, RuntimeExecutionService};
use crate::foreign::{
    ForeignAdapter, ForeignPlatform, ForeignRuntime, PythonAdapter, PythonRuntimeConfiguration,
};
use crate::RuntimeError;

const CALLBACK_PREFIX: &str = "__runmat_python_isolated_callback_";
type ProcessChannel = HostChannel<runmat_process_host::ipc::private_control::BlockingControlWriter>;

pub async fn run_python_extension_host() -> Result<(), String> {
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
    let control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(|error| error.to_string())?;
    let mut reader = control.reader;
    let mut writer = control.writer;
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
    let limits = session.limits;
    let inbox = super::super::super::isolation_input::spawn_input_reader(
        reader,
        "runmat-python-input",
        move |reader| read_python_driver(reader, limits),
    )?;
    let _input_close = super::super::super::isolation_input::InputMailboxClose::new(inbox.clone());
    let channel = Rc::new(RefCell::new(HostChannel {
        inbox,
        writer,
        limits,
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
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let local_identity = context.local_identity();
    let context = context.with_service_ports(
        RuntimeServicePorts::default()
            .with_foreign(foreign)
            .with_call(callback_router.clone())
            .with_local_tasks(supervisor.submission_port(local_identity)),
    );
    let lifecycle = RuntimeLifecycleHost::new(&context, Rc::clone(&supervisor))
        .map_err(|error| error.to_string())?;
    ACTIVE_HOST.with(|host| {
        host.replace(Some((adapter.clone(), context.clone())));
    });

    let outcome = loop {
        let inbox = channel.borrow().inbox.clone();
        let message = supervisor
            .participate_foreground(inbox.next())
            .await
            .map_err(|error| error.to_string())??;
        match message {
            PythonDriverMessage::Invoke(request) => {
                let request_id = request.request_id;
                callback_router.active_request.set(request_id);
                let outcome = invoke_request(
                    &adapter,
                    &context,
                    &supervisor,
                    &lifecycle,
                    request,
                    &snapshots,
                    &resources,
                )
                .await;
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
                // Extension hosts own this handle table, not a source workspace.
                {
                    let mut resources = resources.borrow_mut();
                    resources.release(&releases);
                    resources.values.clear();
                    resources.identities.clear();
                }
                let cleanup = supervisor
                    .participate_foreground(lifecycle.drain_gracefully(&context))
                    .await;
                let outcome = match cleanup {
                    Ok(outcome) => outcome.map_err(runtime_wire_error),
                    Err(error) => Err(runtime_wire_error(error)),
                };
                lifecycle.force_close();
                channel
                    .borrow_mut()
                    .write_host(&PythonHostMessage::Shutdown(PythonShutdownResult {
                        request_id,
                        outcome,
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

async fn invoke_request(
    adapter: &Rc<PythonAdapter>,
    context: &RuntimeContext,
    supervisor: &Rc<RuntimeLocalTaskSupervisor>,
    lifecycle: &RuntimeLifecycleHost,
    request: PythonInvocationRequest,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<Vec<PythonWireValue>, PythonWireError> {
    resources.borrow_mut().release(&request.releases);
    let mut arguments = RootedValuesBuilder::with_capacity(request.arguments.len());
    for value in request.arguments {
        arguments.push_rooted(decode_host_value(value, snapshots, resources)?);
    }
    let arguments = arguments
        .finish()
        .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))?;
    let value = supervisor
        .participate_foreground(async {
            let value = context
                .clone()
                .scope(adapter.invoke(
                    context.clone(),
                    ForeignCall {
                        adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
                        symbol: request.operation,
                        arguments: crate::call::RootedFunctionArguments::ordinary(arguments),
                        requested_outputs: request.requested_outputs as usize,
                    },
                ))
                .await
                .into_invoked_result()?;
            lifecycle.service_safepoint(context)?;
            lifecycle.drain_gracefully(context).await?;
            Ok(value)
        })
        .await
        .map_err(runtime_wire_error)?
        .map_err(runtime_wire_error)?;
    value.with_owned_sequence(|value| encode_host_outputs(value, snapshots, resources))
}

fn encode_host_outputs(
    sequence: runmat_value::ValueSequence,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<Vec<PythonWireValue>, PythonWireError> {
    let values = sequence.into_output_storage();
    if values.len() > super::PYTHON_HOST_MAX_OUTPUTS {
        return Err(wire_error(
            "RunMat:Python:HostProtocol",
            "Python output count exceeds the protocol limit",
        ));
    }
    values
        .try_into_owned()
        .map_err(|_| {
            wire_error(
                "RunMat:Python:HostProtocol",
                "shared output storage requires separate admission",
            )
        })?
        .into_iter()
        .map(|value| encode_host_value(value, snapshots, resources))
        .collect()
}

fn decode_host_value(
    value: PythonWireValue,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<RootedValueMember, PythonWireError> {
    match value {
        PythonWireValue::Portable(transfer) => decode_portable(&transfer, snapshots),
        PythonWireValue::Foreign(reference) => {
            let value = resources
                .borrow()
                .values
                .get(&reference.id)
                .cloned()
                .ok_or_else(|| {
                    wire_error("RunMat:Python:StaleHandle", "Python object was released")
                })?;
            root_plain_value(value)
        }
        PythonWireValue::Callback { id } => {
            root_plain_value(Value::function_handle(format!("{CALLBACK_PREFIX}{id}")))
        }
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
                type_identity: reference.type_identity.clone(),
                ownership: reference.ownership,
                affinity: reference.affinity,
                lifetime: reference.lifetime,
            }))
        }
        Value::OutputList(_) => Err(wire_error(
            "RunMat:TransientSequenceNotPortable",
            "transient output sequences cannot be encoded as nested Python values",
        )),
        other => encode_portable(&other, snapshots).map(PythonWireValue::Portable),
    }
}

struct HostCallbackRouter {
    channel: Rc<RefCell<ProcessChannel>>,
    snapshots: Rc<SharedSnapshotStore>,
    resources: Rc<RefCell<RemoteResources>>,
    active_request: Cell<u64>,
    depth: Cell<u16>,
}

impl RuntimeCallService for HostCallbackRouter {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in self.resources.try_borrow().ok()?.values.values() {
            visitor(value)?;
        }
        Some(())
    }
    fn resolve(&self, _name: &str) -> Option<usize> {
        None
    }

    fn invoke(
        &self,
        runtime: RuntimeContext,
        request: RuntimeCallRequest,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<crate::execution::RootedValueSequence, RuntimeError>>
                + 'static,
        >,
    > {
        let result = self
            .invoke_now(&runtime, request)
            .map_err(super::client::runtime_error);
        Box::pin(async move { result })
    }
}

impl HostCallbackRouter {
    fn invoke_now(
        &self,
        runtime: &RuntimeContext,
        request: RuntimeCallRequest,
    ) -> Result<crate::execution::RootedValueSequence, PythonWireError> {
        let descriptor = request.descriptor();
        let crate::call::descriptor::CallableTarget::Resolved { identity, .. } = &descriptor.target
        else {
            return Err(wire_error(
                "RunMat:Python:CallbackProtocol",
                "isolated Python host cannot invoke a local bound callable by driver callback ID",
            ));
        };
        let callback_id = callback_id(identity.as_identity()).ok_or_else(|| {
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
            let (descriptor, _source_roots, _supplied_roots) = request.into_parts();
            let requested_outputs = descriptor.requested_outputs;
            let arguments = descriptor
                .args
                .into_iter()
                .map(|value| encode_host_value(value, &self.snapshots, &self.resources))
                .collect::<Result<Vec<_>, _>>()?;
            self.channel
                .borrow_mut()
                .write_host(&PythonHostMessage::Callback(PythonCallbackRequest {
                    request_id,
                    callback_id,
                    depth,
                    requested_outputs: u32::try_from(requested_outputs).map_err(|_| {
                        wire_error(
                            "RunMat:Python:CallbackProtocol",
                            "Python callback output count exceeds u32",
                        )
                    })?,
                    arguments,
                }))
                .map_err(|error| wire_error("RunMat:Python:HostTransport", error))?;
            loop {
                let inbox = self.channel.borrow().inbox.clone();
                let message = inbox
                    .read_blocking()
                    .map_err(|error| wire_error("RunMat:Python:HostTransport", error))?;
                match message {
                    PythonDriverMessage::CallbackResult(result)
                        if result.request_id == request_id && result.callback_id == callback_id =>
                    {
                        let returned = result.outcome?;
                        let mut values = RootedValuesBuilder::with_capacity(returned.len());
                        for value in returned {
                            values.push_rooted(decode_host_value(
                                value,
                                &self.snapshots,
                                &self.resources,
                            )?);
                        }
                        let values = values.finish().map_err(|error| {
                            wire_error("RunMat:Python:ValueDecode", error.to_string())
                        })?;
                        return values.with_owned_output_storage(|values| {
                            let sequence = if requested_outputs == 1 && values.len() == 1 {
                                runmat_value::ValueSequence::single(
                                    values
                                        .try_into_first()
                                        .map_err(|_| {
                                            wire_error(
                                                "RunMat:Python:CallbackProtocol",
                                                "shared output storage requires separate admission",
                                            )
                                        })?
                                        .expect("one Python callback output"),
                                )
                            } else {
                                runmat_value::ValueSequence::from_output_storage(values)
                            }
                            .map_err(|error| {
                                wire_error("RunMat:Python:CallbackProtocol", error.to_string())
                            })?;
                            crate::execution::RootedValueSequence::new(sequence).map_err(|error| {
                                wire_error("RunMat:Python:ValueDecode", error.to_string())
                            })
                        });
                    }
                    PythonDriverMessage::Invoke(nested) => {
                        let nested_id = nested.request_id;
                        let adapter = current_python_adapter()?;
                        let outcome = invoke_nested_request(
                            &adapter,
                            // This callback generation owns the interpreter's
                            // reentry mailbox. The bootstrap context cannot
                            // dispatch onto a lane waiting for this callback.
                            runtime,
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

// Python callback dispatch is a synchronous foreign ABI boundary. Nested
// driver invocations use the same ordered input mailbox while the parent
// callback is active; RunMat callback execution belongs to the parent driver.
fn invoke_nested_request(
    adapter: &Rc<PythonAdapter>,
    context: &RuntimeContext,
    request: PythonInvocationRequest,
    snapshots: &SharedSnapshotStore,
    resources: &Rc<RefCell<RemoteResources>>,
) -> Result<Vec<PythonWireValue>, PythonWireError> {
    resources.borrow_mut().release(&request.releases);
    let mut arguments = RootedValuesBuilder::with_capacity(request.arguments.len());
    for value in request.arguments {
        arguments.push_rooted(decode_host_value(value, snapshots, resources)?);
    }
    let arguments = arguments
        .finish()
        .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))?;
    arguments
        .with_unbounded_values(|arguments| {
            let value =
                pollster::block_on(context.scope(adapter.invoke(
                    context.clone(),
                    ForeignCall {
                        adapter: runmat_python::PYTHON_ADAPTER_ID.into(),
                        symbol: request.operation,
                        arguments:
                            crate::call::RootedFunctionArguments::ordinary(
                                crate::execution::RootedValues::new(arguments).map_err(
                                    |error| runtime_wire_error(error.into_runtime_error()),
                                )?,
                            ),
                        requested_outputs: request.requested_outputs as usize,
                    },
                )))
                .into_invoked_result()
                .map_err(runtime_wire_error)?;
            value.with_owned_sequence(|value| encode_host_outputs(value, snapshots, resources))
        })
        .map_err(runtime_wire_error)?
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
                reference.host_identity.clone(),
                reference.handle,
                reference.generation,
            ));
        }
    }
}

struct HostChannel<W> {
    inbox: std::sync::Arc<super::super::super::isolation_input::InputMailbox<PythonDriverMessage>>,
    writer: W,
    limits: FrameLimits,
}

impl<W: Write> HostChannel<W> {
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

fn read_python_driver(
    reader: &mut impl Read,
    limits: FrameLimits,
) -> Result<PythonDriverMessage, String> {
    let payload = read_payload_blocking(reader, limits).map_err(|error| error.to_string())?;
    let message: PythonDriverMessage = serde_json::from_slice(&payload)
        .map_err(|error| format!("invalid Python driver record: {error}"))?;
    message
        .validate()
        .map_err(|error| format!("invalid Python driver record: {error}"))?;
    Ok(message)
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
