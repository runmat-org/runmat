use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use futures::channel::{mpsc as async_mpsc, oneshot};
use futures::{FutureExt, StreamExt};
use runmat_mex::{MexDiagnostic, MexModuleRetirement, MxBoundaryCodec, MxValueContext};
use runmat_process_host::ipc::{
    authenticate_host_blocking, read_payload_blocking, write_payload_blocking, FrameLimits,
    HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;

use super::{
    decode_value_transfer_rooted_value, decode_value_transfers_rooted, encode_value_transfer,
    DriverMessage, HostMessage, MexBinaryTier, MexCallbackCancellation, MexCallbackOperation,
    MexCallbackOutput, MexCallbackRequest, MexCallbackResult, MexInvocationOutput,
    MexInvocationRequest, MexInvocationResult, MexLifecycleOutcome, MexLifecycleResult,
    MexWireDiagnostic, MexWireError, MEX_HOST_MAX_MESSAGE_BYTES, MEX_HOST_PROTOCOL,
    MEX_HOST_SCHEMA_VERSION, MEX_HOST_SECRET_ENV, MEX_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::execution::RootedValue;
use crate::foreign::mex::native_lane::{NativeMexLane, NativeModuleLoadError};
use crate::foreign::MexCallbackAncestry;

mod origin;
use origin::IsolatedOriginBoundaryAdapter;

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

    let control = runmat_process_host::ipc::private_control::connect_host_blocking()
        .map_err(|error| error.to_string())?;
    let mut reader = control.reader;
    let mut writer = control.writer;
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
                    callback_ancestry_depth,
                } => {
                    let outcome = clear(
                        &lane,
                        &mut values,
                        request_id,
                        &module_path,
                        callback_ancestry_depth,
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
                    let retirement = shutdown(
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
                            outcome: retirement.outcome,
                        },
                    ))?;
                    if retirement.complete {
                        return Ok(());
                    }
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
    let ancestry =
        MexCallbackAncestry::from_wire(request.callback_ancestry_depth).ok_or_else(|| {
            wire_error(
                "RunMat:MEX:CallbackDepth",
                "callback ancestry exceeds the protocol limit",
                None,
            )
        })?;
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
    let arguments = decode_value_transfers_rooted(&request.arguments, &snapshots)?;
    let inputs = arguments
        .values()
        .iter()
        .map(|value| value_context.encode(value, mode, metadata.interface))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| wire_error("RunMat:MEX:Conversion", error.to_string(), None))?;
    let host = Rc::new(IsolatedMexHostServices::new(
        request.request_id,
        ancestry,
        writer,
        callback_waiters,
        Rc::clone(&snapshots),
    ));
    let codec = Rc::new(MxBoundaryCodec::new(
        value_context.clone(),
        mode,
        metadata.interface,
    ));
    let services = Rc::new(IsolatedOriginBoundaryAdapter::new(host, codec));
    let invocation = lane
        .invoke(&path, inputs, request.requested_outputs as usize, services)
        .await
        .map_err(|error| wire_error("RunMat:MEX:Invocation", error.to_string(), None))?;
    let mut outputs = Vec::with_capacity(invocation.outputs.len());
    for output in &invocation.outputs {
        let value = value_context
            .decode(output)
            .map_err(|error| wire_error("RunMat:MEX:Conversion", error.to_string(), None))?;
        let value = RootedValue::new(value)
            .map_err(|error| wire_error("RunMat:MEX:ValueRoot", error.to_string(), None))?;
        outputs.push(encode_value_transfer(&value, &snapshots)?);
    }
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
    callback_ancestry_depth: u16,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<MexLifecycleOutcome, MexWireError> {
    let ancestry = MexCallbackAncestry::from_wire(callback_ancestry_depth).ok_or_else(|| {
        wire_error(
            "RunMat:MEX:CallbackDepth",
            "callback ancestry exceeds the protocol limit",
            None,
        )
    })?;
    let path = canonical_module_path(path)?;
    let Some(value_context) = values.get(&path).cloned() else {
        return Ok(MexLifecycleOutcome::Cleared);
    };
    let metadata = lane
        .load_with_policy(&path, false)
        .await
        .map_err(native_load_error)?;
    let host = Rc::new(IsolatedMexHostServices::new(
        request_id,
        ancestry,
        writer,
        callback_waiters,
        snapshots,
    ));
    let codec = Rc::new(MxBoundaryCodec::new(
        value_context,
        metadata.mode,
        metadata.interface,
    ));
    let services = Rc::new(IsolatedOriginBoundaryAdapter::new(host, codec));
    match lane.clear(&path, services).await {
        Ok(true) => {
            values.remove(&path);
            Ok(MexLifecycleOutcome::Cleared)
        }
        Ok(false) => Ok(MexLifecycleOutcome::Retained),
        Err(error) => Err(wire_error("RunMat:MEX:Clear", error.to_string(), None)),
    }
}

struct ShutdownRetirement {
    outcome: Result<MexLifecycleOutcome, MexWireError>,
    complete: bool,
}

async fn shutdown<W: Write + 'static>(
    lane: &NativeMexLane,
    values: &mut HashMap<PathBuf, Rc<MxValueContext>>,
    request_id: u64,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
) -> ShutdownRetirement {
    let loaded = values.keys().map(ToOwned::to_owned).collect::<Vec<_>>();
    let mut first_error = None;
    let mut retained = false;
    for path in loaded {
        let Some(value_context) = values.get(&path).cloned() else {
            continue;
        };
        let metadata = match lane.load_with_policy(&path, false).await {
            Ok(metadata) => metadata,
            Err(error) => {
                retained = true;
                first_error.get_or_insert_with(|| native_load_error(error));
                continue;
            }
        };
        let host = Rc::new(IsolatedMexHostServices::new(
            request_id,
            MexCallbackAncestry::ROOT,
            Rc::clone(&writer),
            Arc::clone(&callback_waiters),
            Rc::clone(&snapshots),
        ));
        let codec = Rc::new(MxBoundaryCodec::new(
            value_context,
            metadata.mode,
            metadata.interface,
        ));
        let services = Rc::new(IsolatedOriginBoundaryAdapter::new(host, codec));
        match lane.shutdown_module(&path, services).await {
            Ok(MexModuleRetirement::Retired(result)) => {
                values.remove(&path);
                if let Err(error) = result {
                    first_error.get_or_insert_with(|| {
                        wire_error("RunMat:MEX:Shutdown", error.to_string(), None)
                    });
                }
            }
            Ok(MexModuleRetirement::Retained(error)) | Err(error) => {
                retained = true;
                first_error.get_or_insert_with(|| {
                    wire_error("RunMat:MEX:Shutdown", error.to_string(), None)
                });
            }
        }
    }
    ShutdownRetirement {
        outcome: if retained {
            Ok(MexLifecycleOutcome::Retained)
        } else {
            Ok(MexLifecycleOutcome::Retired {
                diagnostic: first_error,
            })
        },
        complete: !retained,
    }
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
enum CallbackWaiter {
    Pending(oneshot::Sender<CallbackResponse>),
    Abandoned,
}
type CallbackWaiters = Arc<Mutex<HashMap<CallbackKey, CallbackWaiter>>>;

struct CallbackWaiterLease {
    waiters: CallbackWaiters,
    key: CallbackKey,
}

impl Drop for CallbackWaiterLease {
    fn drop(&mut self) {
        let retired = self
            .waiters
            .lock()
            .expect("callback waiter registry is not poisoned")
            .get_mut(&self.key)
            .map(|entry| std::mem::replace(entry, CallbackWaiter::Abandoned));
        drop(retired);
    }
}

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
                    match waiter {
                        Some(CallbackWaiter::Pending(waiter)) => {
                            let _ = waiter.send(Ok(response));
                        }
                        Some(CallbackWaiter::Abandoned) => {}
                        None => {
                            fail_callback_waiters(
                                &callback_waiters,
                                "driver returned an unknown callback identity".into(),
                            );
                            break;
                        }
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
        if let CallbackWaiter::Pending(waiter) = waiter {
            let _ = waiter.send(Err(error.clone()));
        }
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
    depth: Cell<MexCallbackAncestry>,
    writer: Rc<RefCell<HostWriter<W>>>,
    callback_waiters: CallbackWaiters,
    snapshots: Rc<SharedSnapshotStore>,
}

impl<W: Write> IsolatedMexHostServices<W> {
    fn new(
        request_id: u64,
        ancestry: MexCallbackAncestry,
        writer: Rc<RefCell<HostWriter<W>>>,
        callback_waiters: CallbackWaiters,
        snapshots: Rc<SharedSnapshotStore>,
    ) -> Self {
        Self {
            request_id,
            next_callback_id: Cell::new(1),
            depth: Cell::new(ancestry),
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
    async fn callback(
        &self,
        operation: MexCallbackOperation,
        cancellation: &runmat_execution::cancellation::CancellationToken,
    ) -> Result<MexCallbackOutput, MexDiagnostic> {
        let parent = self.depth.get();
        let depth = parent.child().ok_or_else(|| {
            diagnostic("RunMat:MEX:CallbackDepth", "callback depth limit exceeded")
        })?;
        self.depth.set(depth);
        let callback_id = self.next_callback_id.get();
        let Some(next_callback_id) = callback_id.checked_add(1) else {
            self.depth.set(parent);
            return Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "isolated MEX callback identity space is exhausted",
            ));
        };
        self.next_callback_id.set(next_callback_id);
        struct CallbackDepth<'a>(&'a Cell<MexCallbackAncestry>, MexCallbackAncestry);
        impl Drop for CallbackDepth<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        let _depth = CallbackDepth(&self.depth, parent);
        let result =
            async {
                let (response_tx, response_rx) = oneshot::channel();
                self.callback_waiters
                    .lock()
                    .expect("callback waiter registry is not poisoned")
                    .insert(
                        (self.request_id, callback_id),
                        CallbackWaiter::Pending(response_tx),
                    );
                let _waiter = CallbackWaiterLease {
                    waiters: Arc::clone(&self.callback_waiters),
                    key: (self.request_id, callback_id),
                };
                if let Err(error) = self.writer.borrow_mut().write_host(&HostMessage::Callback(
                    MexCallbackRequest {
                        request_id: self.request_id,
                        callback_id,
                        depth: depth.wire(),
                        operation,
                    },
                )) {
                    return Err(diagnostic("RunMat:MEX:CallbackTransport", error));
                }
                // This counter describes synchronous callback admission, not
                // the number of independently suspended async siblings.
                drop(_depth);
                let response = {
                    let reply = response_rx.fuse();
                    let cancelled = cancellation.cancelled().fuse();
                    futures::pin_mut!(reply, cancelled);
                    futures::select! {
                        result = reply => result,
                        _ = cancelled => {
                            self.writer.borrow_mut().write_host(&HostMessage::CallbackCancelled(
                                MexCallbackCancellation { request_id: self.request_id, callback_id }
                            )).map_err(|error| diagnostic("RunMat:MEX:CallbackTransport", error))?;
                            reply.await
                        }
                    }
                }
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
            }
            .await;
        result
    }
}

fn diagnostic(identifier: &str, message: impl Into<String>) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some(identifier.into()),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::MEX_HOST_MAX_CALLBACK_DEPTH;
    use super::*;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    #[test]
    fn suspended_callback_waiter_retires_exactly_on_cancel_or_reply() {
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let host = IsolatedMexHostServices::new(
            7,
            MexCallbackAncestry::ROOT,
            Rc::new(RefCell::new(HostWriter {
                writer: Vec::<u8>::new(),
                limits: FrameLimits {
                    max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
                },
            })),
            Arc::clone(&waiters),
            Rc::new(SharedSnapshotStore::create().expect("snapshot store")),
        );
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let cancellation = runmat_execution::cancellation::CancellationSource::new().token();
        let mut cancelled = Box::pin(host.callback(
            MexCallbackOperation::Eval {
                command: "cancelled".into(),
            },
            &cancellation,
        ));
        assert!(matches!(cancelled.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(waiters.lock().unwrap().len(), 1);
        let cancelled_id = host.next_callback_id.get() - 1;
        drop(cancelled);
        assert!(matches!(
            waiters.lock().unwrap().remove(&(7, cancelled_id)),
            Some(CallbackWaiter::Abandoned)
        ));
        assert_eq!(host.depth.get(), MexCallbackAncestry::ROOT);

        let mut completed = Box::pin(host.callback(
            MexCallbackOperation::Eval {
                command: "completed".into(),
            },
            &cancellation,
        ));
        assert!(matches!(completed.as_mut().poll(&mut cx), Poll::Pending));
        let callback_id = host.next_callback_id.get() - 1;
        let sender = waiters
            .lock()
            .unwrap()
            .remove(&(7, callback_id))
            .expect("exact waiter");
        let CallbackWaiter::Pending(sender) = sender else {
            panic!("live callback must retain its reply sender");
        };
        sender
            .send(Ok(MexCallbackResult {
                request_id: 7,
                callback_id,
                outcome: Ok(MexCallbackOutput::Unit),
            }))
            .expect("publish callback result");
        assert!(matches!(
            Pin::as_mut(&mut completed).poll(&mut cx),
            Poll::Ready(Ok(MexCallbackOutput::Unit))
        ));
        drop(completed);
        assert!(waiters.lock().unwrap().is_empty());
        assert_eq!(host.depth.get(), MexCallbackAncestry::ROOT);
    }

    #[test]
    fn running_cancellation_is_published_before_exact_terminal_reply() {
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let writer = Rc::new(RefCell::new(HostWriter {
            writer: Vec::<u8>::new(),
            limits: FrameLimits {
                max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
            },
        }));
        let host = IsolatedMexHostServices::new(
            9,
            MexCallbackAncestry::ROOT,
            Rc::clone(&writer),
            Arc::clone(&waiters),
            Rc::new(SharedSnapshotStore::create().expect("snapshot store")),
        );
        let cancellation = CancellationSource::new();
        let cancellation_token = cancellation.token();
        let mut callback = Box::pin(host.callback(
            MexCallbackOperation::Eval {
                command: "suspended".into(),
            },
            &cancellation_token,
        ));
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(callback.as_mut().poll(&mut cx), Poll::Pending));
        let callback_id = host.next_callback_id.get() - 1;
        cancellation.cancel(CancellationReason::User);
        assert!(matches!(callback.as_mut().poll(&mut cx), Poll::Pending));
        assert!(matches!(
            waiters.lock().unwrap().get(&(9, callback_id)),
            Some(CallbackWaiter::Pending(_))
        ));

        let bytes = writer.borrow().writer.clone();
        let mut reader = std::io::Cursor::new(bytes);
        let limits = FrameLimits {
            max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
        };
        let first: HostMessage =
            serde_json::from_slice(&read_payload_blocking(&mut reader, limits).unwrap()).unwrap();
        assert!(matches!(
            first,
            HostMessage::Callback(MexCallbackRequest {
                request_id: 9,
                callback_id: id,
                ..
            }) if id == callback_id
        ));
        let second: HostMessage =
            serde_json::from_slice(&read_payload_blocking(&mut reader, limits).unwrap()).unwrap();
        assert_eq!(
            second,
            HostMessage::CallbackCancelled(MexCallbackCancellation {
                request_id: 9,
                callback_id,
            })
        );

        let sender = waiters.lock().unwrap().remove(&(9, callback_id)).unwrap();
        let CallbackWaiter::Pending(sender) = sender else {
            panic!("cancelled running callback must retain its exact reply waiter");
        };
        sender
            .send(Ok(MexCallbackResult {
                request_id: 9,
                callback_id,
                outcome: Ok(MexCallbackOutput::Unit),
            }))
            .expect("terminal callback reply");
        assert!(matches!(
            callback.as_mut().poll(&mut cx),
            Poll::Ready(Ok(MexCallbackOutput::Unit))
        ));
        drop(callback);
        assert!(waiters.lock().unwrap().is_empty());
    }

    #[test]
    fn suspended_siblings_do_not_consume_logical_callback_depth() {
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let host = IsolatedMexHostServices::new(
            11,
            MexCallbackAncestry::ROOT,
            Rc::new(RefCell::new(HostWriter {
                writer: Vec::<u8>::new(),
                limits: FrameLimits {
                    max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
                },
            })),
            Arc::clone(&waiters),
            Rc::new(SharedSnapshotStore::create().expect("snapshot store")),
        );
        let cancellation = CancellationSource::new().token();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut callbacks = Vec::new();
        for _ in 0..=usize::from(MEX_HOST_MAX_CALLBACK_DEPTH) {
            let mut callback = Box::pin(host.callback(
                MexCallbackOperation::Eval {
                    command: "independent".into(),
                },
                &cancellation,
            ));
            assert!(matches!(callback.as_mut().poll(&mut cx), Poll::Pending));
            callbacks.push(callback);
        }
        assert_eq!(waiters.lock().unwrap().len(), callbacks.len());
        assert_eq!(host.depth.get(), MexCallbackAncestry::ROOT);
        drop(callbacks);
        assert_eq!(host.depth.get(), MexCallbackAncestry::ROOT);
    }

    #[test]
    fn inherited_sixty_third_ancestor_emits_sixty_four_then_rejects_sixty_five() {
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let writer = Rc::new(RefCell::new(HostWriter {
            writer: Vec::<u8>::new(),
            limits: FrameLimits {
                max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
            },
        }));
        let parent = MexCallbackAncestry::from_wire(63).unwrap();
        let host = IsolatedMexHostServices::new(
            21,
            parent,
            Rc::clone(&writer),
            Arc::clone(&waiters),
            Rc::new(SharedSnapshotStore::create().expect("snapshot store")),
        );
        let cancellation = CancellationSource::new().token();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut callback = Box::pin(host.callback(
            MexCallbackOperation::Eval {
                command: "nested".into(),
            },
            &cancellation,
        ));
        assert!(matches!(callback.as_mut().poll(&mut cx), Poll::Pending));
        assert_eq!(host.depth.get(), parent);
        let bytes = writer.borrow().writer.clone();
        let mut reader = std::io::Cursor::new(bytes);
        let first: HostMessage = serde_json::from_slice(
            &read_payload_blocking(
                &mut reader,
                FrameLimits {
                    max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
                },
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(first, HostMessage::Callback(request) if request.depth == 64));
        drop(callback);
        assert_eq!(host.depth.get(), parent);

        host.depth.set(MexCallbackAncestry::from_wire(64).unwrap());
        let published = writer.borrow().writer.len();
        let rejected = futures::executor::block_on(host.callback(
            MexCallbackOperation::Eval {
                command: "never run".into(),
            },
            &cancellation,
        ))
        .expect_err("a 65th callback cannot publish a request");
        assert_eq!(
            rejected.identifier.as_deref(),
            Some("RunMat:MEX:CallbackDepth")
        );
        assert_eq!(writer.borrow().writer.len(), published);
        assert_eq!(host.depth.get().wire(), 64);
        assert!(waiters
            .lock()
            .unwrap()
            .values()
            .all(|waiter| matches!(waiter, CallbackWaiter::Abandoned)));
    }
}
