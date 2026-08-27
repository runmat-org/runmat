use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};

use futures::channel::mpsc as async_mpsc;
use futures::{FutureExt, StreamExt};
use runmat_mex::{DirectMexBoundaryHostServices, MexDiagnostic, MexHostServices, MxValueContext};
use runmat_process_host::ipc::{
    authenticate_host_blocking, read_payload_blocking, write_payload_blocking, FrameLimits,
    HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    decode_value_transfer, encode_value_transfer, DriverMessage, HostMessage, MexBinaryTier,
    MexCallbackOperation, MexCallbackOutput, MexCallbackRequest, MexCallbackResult,
    MexInvocationOutput, MexInvocationRequest, MexInvocationResult, MexLifecycleOutcome,
    MexLifecycleResult, MexWireDiagnostic, MexWireError, MEX_HOST_MAX_CALLBACK_DEPTH,
    MEX_HOST_MAX_MESSAGE_BYTES, MEX_HOST_PROTOCOL, MEX_HOST_SCHEMA_VERSION, MEX_HOST_SECRET_ENV,
    MEX_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::foreign::mex::native_lane::{NativeMexLane, NativeModuleLoadError};

pub async fn run_mex_extension_host() -> Result<(), String> {
    let encoded_secret = std::env::var(MEX_HOST_SECRET_ENV)
        .map_err(|_| "extension host bootstrap secret is unavailable".to_string())?;
    std::env::remove_var(MEX_HOST_SECRET_ENV);
    let secret = SessionSecret::from_hex(&encoded_secret).map_err(|error| error.to_string())?;
    drop(encoded_secret);
    let snapshot_root = std::env::var_os(MEX_HOST_SNAPSHOT_ROOT_ENV)
        .ok_or_else(|| "extension host snapshot root is unavailable".to_string())?;
    std::env::remove_var(MEX_HOST_SNAPSHOT_ROOT_ENV);
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
            MEX_HOST_PROTOCOL,
            MEX_HOST_SCHEMA_VERSION,
            MEX_HOST_MAX_MESSAGE_BYTES,
        ),
        &secret,
    )
    .map_err(|error| error.to_string())?;
    let writer = Rc::new(RefCell::new(HostWriter {
        writer,
        limits: session.limits,
    }));
    let callback_waiters = Arc::new(Mutex::new(HashMap::new()));
    let (driver_tx, mut driver_rx) = async_mpsc::unbounded();
    spawn_driver_reader(
        reader,
        session.limits,
        driver_tx,
        Arc::clone(&callback_waiters),
    )?;
    let lane = Rc::new(NativeMexLane::spawn()?);
    {
        let mut values = HashMap::<PathBuf, Rc<MxValueContext>>::new();
        let mut background_services_available = true;
        loop {
            let message = if background_services_available {
                let driver = driver_rx.next().fuse();
                let background = lane.service_background_once().fuse();
                futures::pin_mut!(driver, background);
                futures::select! {
                    message = driver => message,
                    serviced = background => {
                        background_services_available = serviced;
                        continue;
                    }
                }
            } else {
                driver_rx.next().await
            };
            let Some(message) = message else {
                return Err("extension host driver transport closed".to_string());
            };
            match message {
                DriverMessage::Invoke(request) => {
                    let request_id = request.request_id;
                    let outcome = invoke(
                        &lane,
                        &mut values,
                        request,
                        Rc::clone(&writer),
                        Arc::clone(&callback_waiters),
                        Rc::clone(&snapshots),
                    )
                    .await;
                    writer.borrow_mut().write_host(&HostMessage::Invocation(
                        MexInvocationResult {
                            request_id,
                            outcome,
                        },
                    ))?;
                }
                DriverMessage::Clear {
                    request_id,
                    module_path,
                } => {
                    let outcome = clear(
                        &lane,
                        &mut values,
                        request_id,
                        &module_path,
                        Rc::clone(&writer),
                        Arc::clone(&callback_waiters),
                        Rc::clone(&snapshots),
                    )
                    .await;
                    writer.borrow_mut().write_host(&HostMessage::Lifecycle(
                        MexLifecycleResult {
                            request_id,
                            outcome,
                        },
                    ))?;
                }
                DriverMessage::Shutdown { request_id } => {
                    let outcome = shutdown(
                        &lane,
                        &mut values,
                        request_id,
                        Rc::clone(&writer),
                        Arc::clone(&callback_waiters),
                        Rc::clone(&snapshots),
                    )
                    .await;
                    writer.borrow_mut().write_host(&HostMessage::Lifecycle(
                        MexLifecycleResult {
                            request_id,
                            outcome: outcome.map(|()| MexLifecycleOutcome::Shutdown),
                        },
                    ))?;
                    return Ok(());
                }
                DriverMessage::CallbackResult(_) => {
                    return Err("callback result arrived outside an active callback".into());
                }
            }
        }
    }
}

async fn invoke<W: Write + 'static>(
    lane: &NativeMexLane,
    values: &mut HashMap<PathBuf, Rc<MxValueContext>>,
    request: MexInvocationRequest,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<MexInvocationOutput, MexWireError> {
    request.validate().map_err(protocol_error)?;
    let path = canonical_module_path(&request.module_path)?;
    let compatible_isolated = request.tier == MexBinaryTier::RunMatCompatibleIsolated;
    let metadata = lane
        .load_with_policy(&path, compatible_isolated)
        .await
        .map_err(native_load_error)?;
    let mode = request.api.map_or(metadata.mode, |api| {
        if api.uses_interleaved_complex() {
            runmat_mex::MxApiMode::InterleavedComplex
        } else {
            runmat_mex::MxApiMode::SeparateComplex
        }
    });
    let value_context = values
        .entry(path.clone())
        .or_insert_with(|| Rc::new(MxValueContext::new()))
        .clone();
    let arguments = request
        .arguments
        .iter()
        .map(|value| decode_value_transfer(value, &snapshots))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = arguments
        .iter()
        .map(|value| value_context.encode(value, mode, metadata.interface))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| wire_error("RunMat:MEX:Conversion", error.to_string(), None))?;
    let services = Rc::new(DirectMexBoundaryHostServices::new(
        Rc::new(IsolatedMexHostServices::new(
            request.request_id,
            writer,
            callback_waiters,
            Rc::clone(&snapshots),
        )),
        value_context.clone(),
        mode,
        metadata.interface,
    ));
    let invocation = lane
        .invoke(&path, inputs, request.requested_outputs as usize, services)
        .await
        .map_err(|error| wire_error("RunMat:MEX:Invocation", error.to_string(), None))?;
    let outputs = invocation
        .outputs
        .iter()
        .map(|value| {
            value_context
                .decode(value)
                .map_err(|error| wire_error("RunMat:MEX:Conversion", error.to_string(), None))
                .and_then(|value| encode_value_transfer(&value, &snapshots))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MexInvocationOutput {
        outputs,
        warnings: invocation
            .warnings
            .into_iter()
            .map(|warning| MexWireDiagnostic {
                identifier: warning.identifier,
                message: warning.message,
            })
            .collect(),
        console: invocation.console,
    })
}

async fn clear<W: Write + 'static>(
    lane: &NativeMexLane,
    values: &mut HashMap<PathBuf, Rc<MxValueContext>>,
    request_id: u64,
    path: &str,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<MexLifecycleOutcome, MexWireError> {
    let path = canonical_module_path(path)?;
    let Some(value_context) = values.get(&path).cloned() else {
        return Ok(MexLifecycleOutcome::Cleared);
    };
    let metadata = lane
        .load_with_policy(&path, false)
        .await
        .map_err(native_load_error)?;
    let services = Rc::new(DirectMexBoundaryHostServices::new(
        Rc::new(IsolatedMexHostServices::new(
            request_id,
            writer,
            callback_waiters,
            snapshots,
        )),
        value_context,
        metadata.mode,
        metadata.interface,
    ));
    match lane.clear(&path, services).await {
        Ok(true) => {
            values.remove(&path);
            Ok(MexLifecycleOutcome::Cleared)
        }
        Ok(false) => Ok(MexLifecycleOutcome::Retained),
        Err(error) => Err(wire_error("RunMat:MEX:Clear", error.to_string(), None)),
    }
}

async fn shutdown<W: Write + 'static>(
    lane: &NativeMexLane,
    values: &mut HashMap<PathBuf, Rc<MxValueContext>>,
    request_id: u64,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<(), MexWireError> {
    let loaded = values.drain().collect::<Vec<_>>();
    let mut first_error = None;
    for (path, value_context) in loaded {
        let metadata = match lane.load_with_policy(&path, false).await {
            Ok(metadata) => metadata,
            Err(error) => {
                first_error.get_or_insert_with(|| native_load_error(error));
                continue;
            }
        };
        let services = Rc::new(DirectMexBoundaryHostServices::new(
            Rc::new(IsolatedMexHostServices::new(
                request_id,
                Rc::clone(&writer),
                Arc::clone(&callback_waiters),
                Rc::clone(&snapshots),
            )),
            value_context,
            metadata.mode,
            metadata.interface,
        ));
        if let Err(error) = lane.shutdown_module(&path, services).await {
            first_error
                .get_or_insert_with(|| wire_error("RunMat:MEX:Shutdown", error.to_string(), None));
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn canonical_module_path(path: &str) -> Result<PathBuf, MexWireError> {
    let path = PathBuf::from(path);
    if !path.is_absolute() {
        return Err(wire_error(
            "RunMat:MEX:InvalidModulePath",
            "isolated MEX paths must be absolute",
            None,
        ));
    }
    std::fs::canonicalize(&path).map_err(|error| {
        wire_error(
            "RunMat:MEX:ModuleUnavailable",
            format!("could not resolve '{}': {error}", path.display()),
            Some(path.display().to_string()),
        )
    })
}

fn native_load_error(error: NativeModuleLoadError) -> MexWireError {
    let dependency = match &error {
        NativeModuleLoadError::Failed { dependency, .. } => dependency.clone(),
        NativeModuleLoadError::IsolatedHostRequired => None,
    };
    wire_error("RunMat:MEX:Load", error.to_string(), dependency)
}

fn protocol_error(error: impl std::fmt::Display) -> MexWireError {
    wire_error("RunMat:MEX:HostProtocol", error.to_string(), None)
}

fn wire_error(
    identifier: &str,
    message: impl Into<String>,
    dependency: Option<String>,
) -> MexWireError {
    MexWireError {
        identifier: identifier.into(),
        message: message.into(),
        dependency,
    }
}

type CallbackKey = (u64, u64);
type CallbackResponse = Result<MexCallbackResult, String>;
type CallbackWaiters = Arc<Mutex<HashMap<CallbackKey, mpsc::SyncSender<CallbackResponse>>>>;

struct HostWriter<W> {
    writer: W,
    limits: FrameLimits,
}

impl<W: Write> HostWriter<W> {
    fn write_host(&mut self, message: &HostMessage) -> Result<(), String> {
        message
            .validate()
            .map_err(|error| format!("invalid host record: {error}"))?;
        let payload = serde_json::to_vec(message)
            .map_err(|error| format!("could not encode host record: {error}"))?;
        write_payload_blocking(&mut self.writer, &payload, self.limits)
            .map_err(|error| error.to_string())
    }
}

fn spawn_driver_reader<R: Read + Send + 'static>(
    mut reader: R,
    limits: FrameLimits,
    commands: async_mpsc::UnboundedSender<DriverMessage>,
    callback_waiters: CallbackWaiters,
) -> Result<(), String> {
    std::thread::Builder::new()
        .name("runmat-extension-host-reader".into())
        .spawn(move || loop {
            let result = read_driver(&mut reader, limits);
            match result {
                Ok(DriverMessage::CallbackResult(response)) => {
                    let waiter = callback_waiters
                        .lock()
                        .expect("callback waiter registry is not poisoned")
                        .remove(&(response.request_id, response.callback_id));
                    if let Some(waiter) = waiter {
                        let _ = waiter.send(Ok(response));
                    } else {
                        fail_callback_waiters(
                            &callback_waiters,
                            "driver returned an unknown callback identity".into(),
                        );
                        break;
                    }
                }
                Ok(message) => {
                    if commands.unbounded_send(message).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    fail_callback_waiters(&callback_waiters, error);
                    break;
                }
            }
        })
        .map(|_| ())
        .map_err(|error| format!("could not start the extension-host reader: {error}"))
}

fn fail_callback_waiters(callback_waiters: &CallbackWaiters, error: String) {
    let waiters = std::mem::take(
        &mut *callback_waiters
            .lock()
            .expect("callback waiter registry is not poisoned"),
    );
    for (_, waiter) in waiters {
        let _ = waiter.send(Err(error.clone()));
    }
}

fn read_driver(reader: &mut impl Read, limits: FrameLimits) -> Result<DriverMessage, String> {
    let payload = read_payload_blocking(reader, limits).map_err(|error| error.to_string())?;
    let message: DriverMessage = serde_json::from_slice(&payload)
        .map_err(|error| format!("invalid driver record: {error}"))?;
    message
        .validate()
        .map_err(|error| format!("invalid driver record: {error}"))?;
    Ok(message)
}

struct IsolatedMexHostServices<W: Write> {
    request_id: u64,
    next_callback_id: Cell<u64>,
    depth: Cell<u16>,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
}

impl<W: Write> IsolatedMexHostServices<W> {
    fn new(
        request_id: u64,
        writer: Rc<RefCell<HostWriter<W>>>,
        callback_waiters: CallbackWaiters,
        snapshots: Rc<SharedSnapshotStore>,
    ) -> Self {
        Self {
            request_id,
            next_callback_id: Cell::new(1),
            depth: Cell::new(0),
            writer,
            callback_waiters,
            snapshots,
        }
    }
}

impl<W: Write> Drop for IsolatedMexHostServices<W> {
    fn drop(&mut self) {
        let _ = self
            .writer
            .borrow_mut()
            .write_host(&HostMessage::OriginReleased {
                request_id: self.request_id,
            });
    }
}

impl<W: Write> IsolatedMexHostServices<W> {
    fn callback(
        &self,
        operation: MexCallbackOperation,
    ) -> Result<MexCallbackOutput, MexDiagnostic> {
        let depth = self.depth.get().saturating_add(1);
        if depth > MEX_HOST_MAX_CALLBACK_DEPTH {
            return Err(diagnostic(
                "RunMat:MEX:CallbackDepth",
                "callback depth limit exceeded",
            ));
        }
        self.depth.set(depth);
        let callback_id = self.next_callback_id.get();
        let Some(next_callback_id) = callback_id.checked_add(1) else {
            self.depth.set(depth - 1);
            return Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "isolated MEX callback identity space is exhausted",
            ));
        };
        self.next_callback_id.set(next_callback_id);
        let result =
            (|| {
                let (response_tx, response_rx) = mpsc::sync_channel(1);
                self.callback_waiters
                    .lock()
                    .expect("callback waiter registry is not poisoned")
                    .insert((self.request_id, callback_id), response_tx);
                if let Err(error) = self.writer.borrow_mut().write_host(&HostMessage::Callback(
                    MexCallbackRequest {
                        request_id: self.request_id,
                        callback_id,
                        depth,
                        operation,
                    },
                )) {
                    self.callback_waiters
                        .lock()
                        .expect("callback waiter registry is not poisoned")
                        .remove(&(self.request_id, callback_id));
                    return Err(diagnostic("RunMat:MEX:CallbackTransport", error));
                }
                let response = response_rx
                    .recv()
                    .map_err(|_| {
                        diagnostic(
                            "RunMat:MEX:CallbackTransport",
                            "driver response channel closed",
                        )
                    })?
                    .map_err(|error| diagnostic("RunMat:MEX:CallbackTransport", error))?;
                if response.request_id != self.request_id || response.callback_id != callback_id {
                    return Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "callback result identity does not match the active callback",
                    ));
                }
                response.outcome.map_err(|error| MexDiagnostic {
                    identifier: Some(error.identifier),
                    message: error.message,
                })
            })();
        self.depth.set(depth - 1);
        result
    }
}

impl<W: Write> MexHostServices for IsolatedMexHostServices<W> {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        match self.callback(MexCallbackOperation::Eval {
            command: command.into(),
        })? {
            MexCallbackOutput::Unit => Ok(()),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "eval returned an invalid result",
            )),
        }
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        let arguments = arguments
            .iter()
            .map(|value| encode_value_transfer(value, &self.snapshots))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let requested_outputs = u32::try_from(requested_outputs)
            .map_err(|_| diagnostic("RunMat:MEX:CallbackOutputs", "output count exceeds u32"))?;
        match self.callback(MexCallbackOperation::Call {
            function: function.into(),
            arguments,
            requested_outputs,
        })? {
            MexCallbackOutput::Values(values) => values
                .iter()
                .map(|value| decode_value_transfer(value, &self.snapshots))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| diagnostic(&error.identifier, error.message)),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "call returned an invalid result",
            )),
        }
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        match self.callback(MexCallbackOperation::GetVariable {
            workspace: workspace.into(),
            name: name.into(),
        })? {
            MexCallbackOutput::OptionalValue(value) => value
                .as_ref()
                .map(|value| decode_value_transfer(value, &self.snapshots))
                .transpose()
                .map_err(|error| diagnostic(&error.identifier, error.message)),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "workspace read returned an invalid result",
            )),
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        let value = encode_value_transfer(&value, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        match self.callback(MexCallbackOperation::PutVariable {
            workspace: workspace.into(),
            name: name.into(),
            value,
        })? {
            MexCallbackOutput::Unit => Ok(()),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "workspace write returned an invalid result",
            )),
        }
    }

    fn get_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
    ) -> Result<Value, MexDiagnostic> {
        let object = encode_value_transfer(&object, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let index = u64::try_from(index)
            .map_err(|_| diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64"))?;
        match self.callback(MexCallbackOperation::GetObjectProperty {
            object,
            index,
            name: name.into(),
        })? {
            MexCallbackOutput::Values(values) if values.len() == 1 => {
                decode_value_transfer(&values[0], &self.snapshots)
                    .map_err(|error| diagnostic(&error.identifier, error.message))
            }
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "property read returned an invalid result",
            )),
        }
    }

    fn set_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        let object = encode_value_transfer(&object, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let value = encode_value_transfer(&value, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let index = u64::try_from(index)
            .map_err(|_| diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64"))?;
        match self.callback(MexCallbackOperation::SetObjectProperty {
            object,
            index,
            name: name.into(),
            value,
        })? {
            MexCallbackOutput::Values(values) if values.len() == 1 => {
                decode_value_transfer(&values[0], &self.snapshots)
                    .map_err(|error| diagnostic(&error.identifier, error.message))
            }
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "property write returned an invalid result",
            )),
        }
    }
}

fn diagnostic(identifier: &str, message: impl Into<String>) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some(identifier.into()),
        message: message.into(),
    }
}
