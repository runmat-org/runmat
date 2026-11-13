use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::rc::Rc;

use runmat_process_host::ipc::{
    authenticate_host_blocking, read_payload_blocking, write_payload_blocking, FrameLimits,
    HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    decode_portable, encode_portable, root_plain_value, wire_error, NativeCallbackRequest,
    NativeDriverMessage, NativeHostMessage, NativeInvocationRequest, NativeInvocationResult,
    NativeRemoteReference, NativeShutdownResult, NativeWireError, NativeWireValue,
    NATIVE_FFI_HOST_FRONTEND_ENV, NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH,
    NATIVE_FFI_HOST_MAX_MESSAGE_BYTES, NATIVE_FFI_HOST_PROTOCOL, NATIVE_FFI_HOST_SCHEMA_VERSION,
    NATIVE_FFI_HOST_SECRET_ENV, NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{
    ForeignCall, RuntimeContext, RuntimeLifecycleHost, RuntimeLocalTaskSupervisor,
    RuntimeServicePorts,
};
use crate::execution::{
    RootedValue, RootedValueLease, RootedValueMember, RootedValuesBuilder, RuntimeExecutionService,
};
use crate::foreign::{
    ForeignAdapter, ForeignHandleRegistry, NativeFfiAdapter, NativeFfiCallbackRouter,
};

pub async fn run_native_ffi_extension_host() -> Result<(), String> {
    let encoded_secret = std::env::var(NATIVE_FFI_HOST_SECRET_ENV)
        .map_err(|_| "native-library host bootstrap secret is unavailable".to_string())?;
    std::env::remove_var(NATIVE_FFI_HOST_SECRET_ENV);
    let secret = SessionSecret::from_hex(&encoded_secret).map_err(|error| error.to_string())?;
    drop(encoded_secret);
    let snapshot_root = std::env::var_os(NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV)
        .ok_or_else(|| "native-library host snapshot root is unavailable".to_string())?;
    std::env::remove_var(NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV);
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
            NATIVE_FFI_HOST_PROTOCOL,
            NATIVE_FFI_HOST_SCHEMA_VERSION,
            NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
        ),
        &secret,
    )
    .map_err(|error| error.to_string())?;
    let limits = session.limits;
    let frontend = std::env::var_os(NATIVE_FFI_HOST_FRONTEND_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| "clang".into());
    std::env::remove_var(NATIVE_FFI_HOST_FRONTEND_ENV);
    run_host(reader, writer, limits, snapshots, frontend).await
}

pub(super) fn run_thread_host(
    reader: impl Read + Send + 'static,
    writer: impl Write + 'static,
    resources: &super::client::NativeChildResources,
    frontend: PathBuf,
) -> Result<(), String> {
    let snapshots = Rc::new(
        SharedSnapshotStore::open_existing(resources.root_path())
            .map_err(|error| error.to_string())?,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("could not start native host runtime: {error}"))?;
    runtime.block_on(run_host(
        reader,
        writer,
        FrameLimits {
            max_message_bytes: NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
        },
        snapshots,
        frontend,
    ))
}

async fn run_host(
    reader: impl Read + Send + 'static,
    writer: impl Write + 'static,
    limits: FrameLimits,
    snapshots: Rc<SharedSnapshotStore>,
    frontend: PathBuf,
) -> Result<(), String> {
    let inbox = super::super::super::isolation_input::spawn_input_reader(
        reader,
        "runmat-native-ffi-input",
        move |reader| read_native_driver(reader, limits),
    )?;
    let _input_close = super::super::super::isolation_input::InputMailboxClose::new(inbox.clone());
    let channel = Rc::new(RefCell::new(HostChannel {
        inbox,
        writer: Box::new(writer),
        limits,
    }));
    let callback_router = Rc::new(IsolatedCallbackRouter {
        channel: Rc::clone(&channel),
        snapshots: Rc::clone(&snapshots),
        active_request: Cell::new(0),
        depth: Cell::new(0),
    });
    let handles = ForeignHandleRegistry::default();
    let adapter = NativeFfiAdapter::new_with_callback_router_and_frontend(
        handles,
        Some(callback_router.clone() as Rc<dyn NativeFfiCallbackRouter>),
        frontend,
    )
    .map_err(|error| error.to_string())?;
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let local_identity = context.local_identity();
    let context = context.with_service_ports(
        RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(local_identity)),
    );
    let lifecycle = RuntimeLifecycleHost::new(&context, Rc::clone(&supervisor))
        .map_err(|error| error.to_string())?;
    let mut resources = RemoteResources::default();

    loop {
        let inbox = channel.borrow().inbox.clone();
        let message = supervisor
            .participate_foreground(inbox.next())
            .await
            .map_err(|error| error.to_string())??;
        match message {
            NativeDriverMessage::ProjectLoad(request) => {
                let request_id = request.request_id;
                let outcome = adapter
                    .install_project_load(request)
                    .map_err(runtime_wire_error);
                channel
                    .borrow_mut()
                    .write_host(&NativeHostMessage::Invocation(NativeInvocationResult {
                        request_id,
                        outcome: outcome.map(|()| Vec::new()),
                        user_state_present: adapter.has_user_native_state(),
                    }))?;
            }
            NativeDriverMessage::Invoke(request) => {
                let request_id = request.request_id;
                callback_router.active_request.set(request_id);
                let outcome = invoke(
                    &adapter,
                    &context,
                    &supervisor,
                    &lifecycle,
                    request,
                    &snapshots,
                    &mut resources,
                )
                .await;
                callback_router.active_request.set(0);
                channel
                    .borrow_mut()
                    .write_host(&NativeHostMessage::Invocation(NativeInvocationResult {
                        request_id,
                        outcome,
                        user_state_present: adapter.has_user_native_state(),
                    }))?;
            }
            NativeDriverMessage::Shutdown {
                request_id,
                releases,
            } => {
                // Extension hosts own this handle table, not a source workspace.
                resources.release(&releases);
                resources.values.clear();
                resources.identities.clear();
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
                    .write_host(&NativeHostMessage::Shutdown(NativeShutdownResult {
                        request_id,
                        outcome,
                    }))?;
                return Ok(());
            }
            NativeDriverMessage::CallbackResult(_) => {
                return Err("callback result arrived outside an active callback".into())
            }
        }
    }
}

async fn invoke(
    adapter: &Rc<NativeFfiAdapter>,
    context: &RuntimeContext,
    supervisor: &Rc<RuntimeLocalTaskSupervisor>,
    lifecycle: &RuntimeLifecycleHost,
    request: NativeInvocationRequest,
    snapshots: &SharedSnapshotStore,
    resources: &mut RemoteResources,
) -> Result<Vec<NativeWireValue>, NativeWireError> {
    resources.release(&request.releases);
    let mut arguments = RootedValuesBuilder::with_capacity(request.arguments.len());
    for value in request.arguments {
        arguments.push_rooted(decode_host_value(value, snapshots, resources)?);
    }
    let arguments = arguments
        .finish()
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))?;
    let value = supervisor
        .participate_foreground(async {
            let value = context
                .clone()
                .scope(adapter.invoke(
                    context.clone(),
                    ForeignCall {
                        adapter: runmat_native_ffi::NATIVE_FFI_ADAPTER_ID.into(),
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
    resources: &mut RemoteResources,
) -> Result<Vec<NativeWireValue>, NativeWireError> {
    let values = sequence.into_output_storage();
    if values.len() > super::NATIVE_FFI_HOST_MAX_OUTPUTS {
        return Err(wire_error(
            "RunMat:NativeFFI:HostProtocol",
            "native FFI output count exceeds the protocol limit",
        ));
    }
    values
        .try_into_owned()
        .map_err(|_| {
            wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "shared output storage requires separate admission",
            )
        })?
        .into_iter()
        .map(|value| encode_host_value(value, snapshots, resources))
        .collect()
}

fn decode_host_value(
    value: NativeWireValue,
    snapshots: &SharedSnapshotStore,
    resources: &RemoteResources,
) -> Result<RootedValueMember, NativeWireError> {
    match value {
        NativeWireValue::Portable(transfer) => decode_portable(&transfer, snapshots),
        NativeWireValue::Foreign(reference) => {
            let value = resources
                .values
                .get(&reference.id)
                .cloned()
                .ok_or_else(|| {
                    wire_error(
                        "RunMat:NativeFFI:StaleHandle",
                        "native pointer has been released by the isolated host",
                    )
                })?;
            root_plain_value(value)
        }
        NativeWireValue::Callback { id } => {
            root_plain_value(super::super::isolated_callback_value(id))
        }
    }
}

fn encode_host_value(
    value: Value,
    snapshots: &SharedSnapshotStore,
    resources: &mut RemoteResources,
) -> Result<NativeWireValue, NativeWireError> {
    match value {
        Value::Foreign(reference) => {
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
                    wire_error(
                        "RunMat:NativeFFI:HostProtocol",
                        "remote resource identity exhausted",
                    )
                })?;
                resources.identities.insert(identity, id);
                resources
                    .values
                    .insert(id, Value::Foreign(reference.clone()));
                id
            };
            Ok(NativeWireValue::Foreign(NativeRemoteReference {
                id,
                type_identity: reference.type_identity.clone(),
                ownership: reference.ownership,
                affinity: reference.affinity,
                lifetime: reference.lifetime,
            }))
        }
        Value::OutputList(_) => Err(wire_error(
            "RunMat:TransientSequenceNotPortable",
            "transient output sequences cannot be encoded as nested native FFI values",
        )),
        other => encode_portable(&other, snapshots).map(NativeWireValue::Portable),
    }
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

struct HostChannel {
    inbox: std::sync::Arc<super::super::super::isolation_input::InputMailbox<NativeDriverMessage>>,
    writer: Box<dyn Write>,
    limits: FrameLimits,
}

impl HostChannel {
    fn write_host(&mut self, message: &NativeHostMessage) -> Result<(), String> {
        message
            .validate()
            .map_err(|error| format!("invalid native-library host record: {error}"))?;
        let payload = serde_json::to_vec(message)
            .map_err(|error| format!("could not encode native-library host record: {error}"))?;
        write_payload_blocking(&mut self.writer, &payload, self.limits)
            .map_err(|error| error.to_string())
    }
}

fn read_native_driver(
    reader: &mut impl Read,
    limits: FrameLimits,
) -> Result<NativeDriverMessage, String> {
    let payload = read_payload_blocking(reader, limits).map_err(|error| error.to_string())?;
    let message: NativeDriverMessage = serde_json::from_slice(&payload)
        .map_err(|error| format!("invalid native-library driver record: {error}"))?;
    message
        .validate()
        .map_err(|error| format!("invalid native-library driver record: {error}"))?;
    Ok(message)
}

struct IsolatedCallbackRouter {
    channel: Rc<RefCell<HostChannel>>,
    snapshots: Rc<SharedSnapshotStore>,
    active_request: Cell<u64>,
    depth: Cell<u16>,
}

impl NativeFfiCallbackRouter for IsolatedCallbackRouter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // Channel and snapshot storage contain serialized payloads only.
        Some(())
    }
    fn dispatch(
        &self,
        callback_id: u64,
        arguments: &[Value],
    ) -> Result<RootedValue, crate::RuntimeError> {
        let _input_roots = RootedValueLease::for_values(arguments.iter())
            .map_err(|error| callback_protocol_error(error.to_string()))?;
        let request_id = self.active_request.get();
        if request_id == 0 {
            return Err(callback_protocol_error(
                "native callback has no active invocation",
            ));
        }
        let depth = self.depth.get().saturating_add(1);
        if depth > NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH {
            return Err(callback_protocol_error(
                "native callback depth limit exceeded",
            ));
        }
        self.depth.set(depth);
        let result = (|| {
            let arguments = arguments
                .iter()
                .map(|value| encode_portable(value, &self.snapshots).map(NativeWireValue::Portable))
                .collect::<Result<Vec<_>, _>>()
                .map_err(NativeWireError::into_runtime)?;
            self.channel
                .borrow_mut()
                .write_host(&NativeHostMessage::Callback(NativeCallbackRequest {
                    request_id,
                    callback_id,
                    depth,
                    arguments,
                }))
                .map_err(callback_protocol_error)?;
            let inbox = self.channel.borrow().inbox.clone();
            let response = inbox.read_blocking().map_err(callback_protocol_error)?;
            let NativeDriverMessage::CallbackResult(response) = response else {
                return Err(callback_protocol_error(
                    "native host expected a callback result",
                ));
            };
            if response.request_id != request_id || response.callback_id != callback_id {
                return Err(callback_protocol_error(
                    "native callback result identity does not match the request",
                ));
            }
            let values = response.outcome.map_err(NativeWireError::into_runtime)?;
            let [value] = values.as_slice() else {
                return Err(callback_protocol_error(
                    "native callback did not return exactly one value",
                ));
            };
            match value {
                NativeWireValue::Portable(transfer) => decode_portable(transfer, &self.snapshots)
                    .map_err(NativeWireError::into_runtime)?
                    .into_rooted_value()
                    .map_err(|error| callback_protocol_error(error.to_string())),
                _ => Err(callback_protocol_error(
                    "native callback returned a nonportable value",
                )),
            }
        })();
        self.depth.set(depth - 1);
        result
    }
}

fn runtime_wire_error(error: crate::RuntimeError) -> NativeWireError {
    NativeWireError::from_runtime(error)
}

fn callback_protocol_error(message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:NativeFFI:CallbackProtocol")
        .build()
}
