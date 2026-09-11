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
    decode_portable, encode_portable, wire_error, NativeCallbackRequest, NativeDriverMessage,
    NativeHostMessage, NativeInvocationRequest, NativeInvocationResult, NativeRemoteReference,
    NativeShutdownResult, NativeWireError, NativeWireValue, NATIVE_FFI_HOST_FRONTEND_ENV,
    NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH, NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
    NATIVE_FFI_HOST_PROTOCOL, NATIVE_FFI_HOST_SCHEMA_VERSION, NATIVE_FFI_HOST_SECRET_ENV,
    NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{ForeignCall, RuntimeContext};
use crate::execution::RuntimeExecutionService;
use crate::foreign::{
    ForeignAdapter, ForeignHandleRegistry, NativeFfiAdapter, NativeFfiCallbackRouter,
};

type StdioChannel = HostChannel<std::io::Stdin, std::io::Stdout>;

pub fn run_native_ffi_extension_host() -> Result<(), String> {
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
    let mut reader = std::io::stdin();
    let mut writer = std::io::stdout();
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
    let channel = Rc::new(RefCell::new(HostChannel {
        reader,
        writer,
        limits: session.limits,
    }));
    let callback_router = Rc::new(IsolatedCallbackRouter {
        channel: Rc::clone(&channel),
        snapshots: Rc::clone(&snapshots),
        active_request: Cell::new(0),
        depth: Cell::new(0),
    });
    let handles = ForeignHandleRegistry::default();
    let frontend = std::env::var_os(NATIVE_FFI_HOST_FRONTEND_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| "clang".into());
    std::env::remove_var(NATIVE_FFI_HOST_FRONTEND_ENV);
    let adapter = NativeFfiAdapter::new_with_callback_router_and_frontend(
        handles,
        Some(callback_router.clone() as Rc<dyn NativeFfiCallbackRouter>),
        frontend,
    )
    .map_err(|error| error.to_string())?;
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let mut resources = RemoteResources::default();

    loop {
        let message = channel.borrow_mut().read_driver()?;
        match message {
            NativeDriverMessage::Invoke(request) => {
                let request_id = request.request_id;
                callback_router.active_request.set(request_id);
                let outcome = invoke(&adapter, &context, request, &snapshots, &mut resources);
                callback_router.active_request.set(0);
                channel
                    .borrow_mut()
                    .write_host(&NativeHostMessage::Invocation(NativeInvocationResult {
                        request_id,
                        outcome,
                    }))?;
            }
            NativeDriverMessage::Shutdown {
                request_id,
                releases,
            } => {
                resources.release(&releases);
                channel
                    .borrow_mut()
                    .write_host(&NativeHostMessage::Shutdown(NativeShutdownResult {
                        request_id,
                        outcome: Ok(()),
                    }))?;
                return Ok(());
            }
            NativeDriverMessage::CallbackResult(_) => {
                return Err("callback result arrived outside an active callback".into())
            }
        }
    }
}

fn invoke(
    adapter: &Rc<NativeFfiAdapter>,
    context: &RuntimeContext,
    request: NativeInvocationRequest,
    snapshots: &SharedSnapshotStore,
    resources: &mut RemoteResources,
) -> Result<Vec<NativeWireValue>, NativeWireError> {
    resources.release(&request.releases);
    let arguments = request
        .arguments
        .into_iter()
        .map(|value| decode_host_value(value, snapshots, resources))
        .collect::<Result<Vec<_>, _>>()?;
    let value = pollster::block_on(context.scope(adapter.invoke(
        context.clone(),
        ForeignCall {
            adapter: runmat_native_ffi::NATIVE_FFI_ADAPTER_ID.into(),
            symbol: request.operation,
            arguments,
            requested_outputs: request.requested_outputs as usize,
        },
    )))
    .map_err(runtime_wire_error)?;
    encode_host_outputs(value, snapshots, resources)
}

fn encode_host_outputs(
    sequence: runmat_value::ValueSequence,
    snapshots: &SharedSnapshotStore,
    resources: &mut RemoteResources,
) -> Result<Vec<NativeWireValue>, NativeWireError> {
    let values = sequence.into_values();
    if values.len() > super::NATIVE_FFI_HOST_MAX_OUTPUTS {
        return Err(wire_error(
            "RunMat:NativeFFI:HostProtocol",
            "native FFI output count exceeds the protocol limit",
        ));
    }
    values
        .into_iter()
        .map(|value| encode_host_value(value, snapshots, resources))
        .collect()
}

fn decode_host_value(
    value: NativeWireValue,
    snapshots: &SharedSnapshotStore,
    resources: &RemoteResources,
) -> Result<Value, NativeWireError> {
    match value {
        NativeWireValue::Portable(transfer) => decode_portable(&transfer, snapshots),
        NativeWireValue::Foreign(reference) => {
            resources.values.get(&reference.id).cloned().ok_or_else(|| {
                wire_error(
                    "RunMat:NativeFFI:StaleHandle",
                    "native pointer has been released by the isolated host",
                )
            })
        }
        NativeWireValue::Callback { id } => Ok(super::super::isolated_callback_value(id)),
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
                type_identity: reference.type_identity,
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
    fn read_driver(&mut self) -> Result<NativeDriverMessage, String> {
        let payload = read_payload_blocking(&mut self.reader, self.limits)
            .map_err(|error| error.to_string())?;
        let message: NativeDriverMessage = serde_json::from_slice(&payload)
            .map_err(|error| format!("invalid native-library driver record: {error}"))?;
        message
            .validate()
            .map_err(|error| format!("invalid native-library driver record: {error}"))?;
        Ok(message)
    }

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

struct IsolatedCallbackRouter {
    channel: Rc<RefCell<StdioChannel>>,
    snapshots: Rc<SharedSnapshotStore>,
    active_request: Cell<u64>,
    depth: Cell<u16>,
}

impl NativeFfiCallbackRouter for IsolatedCallbackRouter {
    fn dispatch(&self, callback_id: u64, arguments: &[Value]) -> Result<Value, String> {
        let request_id = self.active_request.get();
        if request_id == 0 {
            return Err("native callback has no active invocation".into());
        }
        let depth = self.depth.get().saturating_add(1);
        if depth > NATIVE_FFI_HOST_MAX_CALLBACK_DEPTH {
            return Err("native callback depth limit exceeded".into());
        }
        self.depth.set(depth);
        let result = (|| {
            let arguments = arguments
                .iter()
                .map(|value| encode_portable(value, &self.snapshots).map(NativeWireValue::Portable))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| error.message)?;
            let mut channel = self.channel.borrow_mut();
            channel.write_host(&NativeHostMessage::Callback(NativeCallbackRequest {
                request_id,
                callback_id,
                depth,
                arguments,
            }))?;
            let response = channel.read_driver()?;
            let NativeDriverMessage::CallbackResult(response) = response else {
                return Err("native host expected a callback result".into());
            };
            if response.request_id != request_id || response.callback_id != callback_id {
                return Err("native callback result identity does not match the request".into());
            }
            let values = response.outcome.map_err(|error| error.message)?;
            let [value] = values.as_slice() else {
                return Err("native callback did not return exactly one value".into());
            };
            match value {
                NativeWireValue::Portable(transfer) => {
                    decode_portable(transfer, &self.snapshots).map_err(|error| error.message)
                }
                _ => Err("native callback returned a nonportable value".into()),
            }
        })();
        self.depth.set(depth - 1);
        result
    }
}

fn runtime_wire_error(error: crate::RuntimeError) -> NativeWireError {
    NativeWireError {
        identifier: error
            .identifier()
            .unwrap_or("RunMat:NativeFFI:InvocationFailed")
            .into(),
        message: error.to_string(),
    }
}
