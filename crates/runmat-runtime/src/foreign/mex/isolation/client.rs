use std::cell::Cell;
use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use futures::channel::mpsc as async_mpsc;
use futures::StreamExt;
use runmat_mex::{MexApi, MexCallOutput, MexDiagnostic, MexHostServices};
use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::private_control::PrivateControlListener;
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake,
    PendingFrameWrite, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_process_host::{ChildProcess, HiddenMode, HostCommand, ProcessGroupPlacement};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{
    decode_value_transfer_rooted_value, decode_value_transfers_rooted, encode_value_transfer,
    DriverMessage, HostMessage, MexBinaryTier, MexCallbackCancellation, MexCallbackOperation,
    MexCallbackOutput, MexCallbackRequest, MexCallbackResult, MexInvocationRequest,
    MexLifecycleOutcome, MexWireError, MEX_HOST_MAX_MESSAGE_BYTES, MEX_HOST_PROTOCOL,
    MEX_HOST_SCHEMA_VERSION, MEX_HOST_SECRET_ENV, MEX_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::foreign::isolation_io::FramedWriter;
use crate::foreign::isolation_output::{self, Completion, HostOutput};
use crate::foreign::mex::session::wire_runtime_error;
use crate::foreign::mex::RootedMexInvocation;
use crate::foreign::RuntimeMexHostServices;
use crate::RuntimeError;

fn root_published_value(output: Box<dyn MexCallOutput>) -> Result<RootedValue, MexWireError> {
    let mut rooted = None;
    output
        .publish(&mut |value| {
            if rooted.is_some() {
                return Err(MexDiagnostic {
                    identifier: Some("RunMat:MEX:CallbackProtocol".into()),
                    message: "single-value callback published multiple values".into(),
                });
            }
            rooted = Some(RootedValue::new(value).map_err(|error| MexDiagnostic {
                identifier: Some("RunMat:MEX:ValueRoot".into()),
                message: error.to_string(),
            })?);
            Ok(())
        })
        .map_err(diagnostic_error)?;
    rooted.ok_or_else(|| {
        wire_error(
            "RunMat:MEX:CallbackProtocol",
            "single-value callback published no value",
            None,
        )
    })
}

const DEFAULT_LIFECYCLE_TIMEOUT: Duration = Duration::from_secs(30);

mod callback_tasks;
mod application_output;
use callback_tasks::ClientCallbackTasks;
mod reply_publication;
use reply_publication::callback_reply_frame;

enum ClientEvent {
    Message(Result<HostMessage, MexWireError>),
    Callback(MexCallbackResult),
}

struct BackgroundReady(Arc<tokio::sync::Notify>);

impl Wake for BackgroundReady {
    fn wake(self: Arc<Self>) {
        self.0.notify_one();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.notify_one();
    }
}

fn poll_background_once<F: Future>(future: F, ready: &Arc<tokio::sync::Notify>) -> Poll<F::Output> {
    let waker = Waker::from(Arc::new(BackgroundReady(Arc::clone(ready))));
    let mut context = Context::from_waker(&waker);
    futures::pin_mut!(future);
    future.poll(&mut context)
}

pub struct IsolatedMexClient {
    module: PathBuf,
    child: ChildProcess,
    snapshots: Rc<SharedSnapshotStore>,
    messages: async_mpsc::UnboundedReceiver<Result<HostMessage, MexWireError>>,
    writer: FramedWriter,
    output: HostOutput,
    limits: FrameLimits,
    next_request_id: u64,
    origins: HashMap<u64, RuntimeContext>,
    callbacks: ClientCallbackTasks,
    pending_reply: Option<PendingFrameWrite>,
    terminal_error: Option<MexWireError>,
    tree_retired: bool,
    background_ready: Arc<tokio::sync::Notify>,
    pending_invocation: Option<(u64, RuntimeContext)>,
    pending_lifecycle: Option<PendingLifecycle>,
}

#[derive(Clone)]
enum PendingLifecycle {
    Clear {
        request_id: u64,
        runtime: RuntimeContext,
        deadline: tokio::time::Instant,
    },
    Shutdown {
        request_id: u64,
        runtime: RuntimeContext,
        deadline: tokio::time::Instant,
    },
    ShutdownExit {
        diagnostic: Option<MexWireError>,
        deadline: tokio::time::Instant,
    },
}

/// Origin-side evidence for shutdown, independent of lifecycle diagnostics.
pub(in crate::foreign::mex) enum IsolatedMexRetirement<E = RuntimeError> {
    Retained(E),
    Retired(Result<(), E>),
}

thread_local! {
    static CALLBACK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

struct CallbackScope;

impl CallbackScope {
    fn enter() -> Self {
        CALLBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

impl Drop for CallbackScope {
    fn drop(&mut self) {
        CALLBACK_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

pub(in crate::foreign::mex) fn in_isolated_callback() -> bool {
    CALLBACK_DEPTH.with(|depth| depth.get() != 0)
}

impl IsolatedMexClient {
    pub async fn spawn(
        runtime: Option<&RuntimeContext>,
        module: PathBuf,
        ready: Arc<tokio::sync::Notify>,
        process_group_placement: ProcessGroupPlacement,
    ) -> Result<Self, MexWireError> {
        let executable = std::env::current_exe()
            .map_err(|error| wire_error("RunMat:MEX:HostExecutable", error.to_string(), None))?;
        let secret = SessionSecret::generate();
        let snapshots =
            Rc::new(SharedSnapshotStore::create().map_err(|error| {
                wire_error("RunMat:MEX:ValueSnapshot", error.to_string(), None)
            })?);
        let mut command = HostCommand::new(executable);
        command.process_group_placement = process_group_placement;
        command.arguments = vec![HiddenMode::ExtensionHost.marker().into()];
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        command
            .environment
            .insert(MEX_HOST_SECRET_ENV.into(), secret.expose_hex());
        let snapshot_root = snapshots.root_path().to_str().ok_or_else(|| {
            wire_error(
                "RunMat:MEX:ValueSnapshot",
                "shared snapshot root is not valid Unicode",
                None,
            )
        })?;
        command
            .environment
            .insert(MEX_HOST_SNAPSHOT_ROOT_ENV.into(), snapshot_root.into());
        let control = PrivateControlListener::bind()
            .map_err(|error| wire_error("RunMat:MEX:HostTransport", error.to_string(), None))?;
        command.stdio =
            runmat_process_host::StdioPolicy::PrivateControl(control.endpoint().clone());
        let mut child = command
            .spawn()
            .await
            .map_err(|error| wire_error("RunMat:MEX:HostSpawn", error.to_string(), None))?;
        let output = child
            .take_application_output()
            .map_err(|error| wire_error("RunMat:MEX:HostTransport", error.to_string(), None))?;
        let control = tokio::time::timeout(DEFAULT_LIFECYCLE_TIMEOUT, control.accept())
            .await
            .map_err(|_| {
                wire_error(
                    "RunMat:MEX:Timeout",
                    "MEX host control bootstrap timed out",
                    None,
                )
            })?
            .map_err(|error| wire_error("RunMat:MEX:HostTransport", error.to_string(), None))?;
        let mut reader = control.reader;
        let mut writer = FramedWriter::Private(control.writer);
        let session = authenticate_driver(
            &mut reader,
            &mut writer,
            HostHandshake::new(
                MEX_HOST_PROTOCOL,
                MEX_HOST_SCHEMA_VERSION,
                MEX_HOST_MAX_MESSAGE_BYTES,
            ),
            &secret,
        )
        .await
        .map_err(|error| wire_error("RunMat:MEX:HostAuthentication", error.to_string(), None))?;
        let limits = session.limits;
        let (message_tx, messages) = async_mpsc::unbounded();
        let background_ready = Arc::clone(&ready);
        tokio::spawn(async move {
            loop {
                let message = read_message(&mut reader, limits).await;
                let terminal = message.is_err();
                if message_tx.unbounded_send(message).is_err() {
                    break;
                }
                ready.notify_one();
                if terminal {
                    break;
                }
            }
        });
        Ok(Self {
            module,
            child,
            snapshots,
            messages,
            writer,
            output: HostOutput::attached(output, runtime)
                .map_err(|error| wire_error("RunMat:MEX:HostTransport", error.to_string(), None))?,
            limits,
            next_request_id: 1,
            origins: HashMap::new(),
            callbacks: ClientCallbackTasks::default(),
            pending_reply: None,
            terminal_error: None,
            tree_retired: false,
            background_ready,
            pending_invocation: None,
            pending_lifecycle: None,
        })
    }

    async fn invoke_protocol(
        &mut self,
        arguments: &crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        api: Option<MexApi>,
        tier: MexBinaryTier,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<RootedMexInvocation, MexWireError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        self.output.select_origin(Some(&runtime));
        let request_id = self.take_request_id()?;
        let mut transferred_arguments = Vec::with_capacity(arguments.len());
        for value in arguments.iter() {
            let host_value = crate::gather_if_needed_async(value)
                .await
                .map_err(|error| wire_error("RunMat:MEX:ValueGather", error.to_string(), None))?;
            transferred_arguments.push(encode_value_transfer(&host_value, &self.snapshots)?);
        }
        let request = DriverMessage::Invoke(MexInvocationRequest {
            request_id,
            callback_ancestry_depth: runtime.mex_callback_ancestry().wire(),
            module_path: self.module.display().to_string(),
            tier,
            api,
            arguments: transferred_arguments,
            requested_outputs: u32::try_from(requested_outputs).map_err(|_| {
                wire_error(
                    "RunMat:MEX:OutputCount",
                    "requested output count exceeds u32",
                    None,
                )
            })?,
        });
        self.origins.insert(request_id, runtime.clone());
        // Record the request before writing: dropping the foreground during a
        // partial write cannot prove the host did not receive the command.
        self.pending_invocation = Some((request_id, runtime.clone()));
        self.write(&request).await?;
        let wait = self.wait_for_invocation(request_id, runtime.clone(), true);
        let outcome = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, wait).await {
                Ok(outcome) => outcome,
                Err(_) => {
                    if self.child.terminate_tree().await.is_ok() {
                        self.pending_invocation = None;
                    }
                    return Err(wire_error(
                        "RunMat:MEX:Timeout",
                        format!(
                            "isolated MEX invocation exceeded {} ms",
                            timeout.as_millis()
                        ),
                        Some(self.module.display().to_string()),
                    ));
                }
            }
        } else {
            wait.await
        };
        if outcome.as_ref().is_err_and(|error| {
            matches!(
                error.identifier.as_str(),
                "RunMat:MEX:Timeout" | "RunMat:MEX:Cancelled" | "RunMat:MEX:HostCrashed"
            )
        }) && self.child.terminate_tree().await.is_ok()
        {
            self.pending_invocation = None;
        }
        outcome
    }

    /// Recover exact protocol alignment after a foreground future was
    /// cancelled. The session retains this client; its next operation or
    /// shutdown must consume the original reply and service its callbacks.
    pub(super) async fn settle_pending(&mut self) -> Result<(), RuntimeError> {
        let Some((request_id, runtime)) = self.pending_invocation.clone() else {
            return Ok(());
        };
        // The abandoned result stays within the codec's producer roots until
        // discarded; it is never republished as a raw managed value.
        let output = self.output.clone();
        let _ = isolation_output::drive(
            &output,
            Some(&runtime),
            Completion::AvailableOutput,
            async {
                self.wait_for_invocation(request_id, runtime.clone(), false)
                    .await
                    .map_err(wire_runtime_error)
            },
        )
        .await?;
        Ok(())
    }

    async fn wait_for_invocation(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        honor_cancellation: bool,
    ) -> Result<RootedMexInvocation, MexWireError> {
        loop {
            if honor_cancellation && runtime.is_cancelled() {
                return Err(wire_error(
                    "RunMat:MEX:Cancelled",
                    "isolated MEX invocation was cancelled",
                    Some(self.module.display().to_string()),
                ));
            }
            // A background callback reply may have been partially published
            // before this foreground owner acquired the client.
            self.flush_pending_reply().await?;
            let have_callbacks = self.callbacks.has_pending();
            let event = tokio::select! {
                message = self.messages.next() => ClientEvent::Message(message.unwrap_or_else(|| Err(wire_error("RunMat:MEX:HostTransport", "host message channel closed", None)))),
                callback = self.callbacks.next(), if have_callbacks => ClientEvent::Callback(callback.expect("active callback task")),
                _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
            };
            let message = match event {
                ClientEvent::Callback(response) => {
                    self.callbacks.finish(&response)?;
                    self.publish_callback_result(response).await?;
                    continue;
                }
                ClientEvent::Message(message) => message,
            };
            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.transport_or_crash(error).await),
            };
            match message {
                HostMessage::Invocation(result) if result.request_id == request_id => {
                    self.pending_invocation = None;
                    let output = result.outcome?;
                    let outputs = decode_value_transfers_rooted(&output.outputs, &self.snapshots)?;
                    return Ok(RootedMexInvocation {
                        outputs,
                        warnings: output
                            .warnings
                            .into_iter()
                            .map(|warning| MexDiagnostic {
                                identifier: warning.identifier,
                                message: warning.message,
                            })
                            .collect(),
                        console: output.console,
                    });
                }
                HostMessage::Callback(callback) => {
                    let origin =
                        self.origins
                            .get(&callback.request_id)
                            .cloned()
                            .ok_or_else(|| {
                                wire_error(
                                    "RunMat:MEX:CallbackProtocol",
                                    "callback origin is no longer active",
                                    None,
                                )
                            })?;
                    self.callbacks
                        .start(callback, origin, Rc::clone(&self.snapshots))?;
                }
                HostMessage::CallbackCancelled(cancelled) => {
                    self.callbacks.cancel(cancelled)?;
                }
                HostMessage::OriginReleased { request_id } => {
                    self.origins.remove(&request_id);
                    self.callbacks.release_request(request_id);
                }
                _ => {
                    return Err(wire_error(
                        "RunMat:MEX:HostProtocol",
                        "host record does not match the active invocation",
                        None,
                    ));
                }
            }
        }
    }

    async fn clear_protocol(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        match self.pending_lifecycle.clone() {
            Some(PendingLifecycle::Clear {
                request_id,
                runtime,
                deadline,
            }) => self.finish_clear(request_id, runtime, deadline).await,
            Some(_) => Err(wire_error(
                "RunMat:MEX:Lifecycle",
                "cannot clear while host shutdown is awaiting retirement",
                Some(self.module.display().to_string()),
            )),
            None => {
                self.output.select_origin(Some(&runtime));
                let request_id = self.take_request_id()?;
                let deadline =
                    tokio::time::Instant::now() + timeout.unwrap_or(DEFAULT_LIFECYCLE_TIMEOUT);
                // A partial write may already have submitted the command.
                self.pending_lifecycle = Some(PendingLifecycle::Clear {
                    request_id,
                    runtime: runtime.clone(),
                    deadline,
                });
                self.write(&DriverMessage::Clear {
                    request_id,
                    module_path: self.module.display().to_string(),
                    callback_ancestry_depth: runtime.mex_callback_ancestry().wire(),
                })
                .await?;
                self.finish_clear(request_id, runtime, deadline).await
            }
        }
    }

    async fn finish_clear(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        deadline: tokio::time::Instant,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        self.output.select_origin(Some(&runtime));
        let outcome = self
            .wait_for_lifecycle_until(request_id, runtime, deadline, "clear")
            .await?;
        self.pending_lifecycle = None;
        Ok(outcome)
    }

    async fn shutdown_protocol(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> IsolatedMexRetirement<MexWireError> {
        if let Some(error) = self.terminal_error.clone() {
            if self.retire_poisoned_host().await.is_ok() {
                return IsolatedMexRetirement::Retired(Err(error));
            }
            return IsolatedMexRetirement::Retained(error);
        }
        if let Some(PendingLifecycle::Clear {
            request_id,
            runtime,
            deadline,
        }) = self.pending_lifecycle.clone()
        {
            if let Err(error) = self.finish_clear(request_id, runtime, deadline).await {
                return self.retirement_after_error(error).await;
            }
        }
        if let Some(PendingLifecycle::ShutdownExit {
            diagnostic,
            deadline,
        }) = self.pending_lifecycle.clone()
        {
            return self.finish_shutdown_exit(diagnostic, deadline).await;
        }
        if let Some(PendingLifecycle::Shutdown {
            request_id,
            runtime,
            deadline,
        }) = self.pending_lifecycle.clone()
        {
            return self.finish_shutdown(request_id, runtime, deadline).await;
        }
        let request_id = match self.take_request_id() {
            Ok(request_id) => request_id,
            Err(error) => return IsolatedMexRetirement::Retained(error),
        };
        self.output.select_origin(Some(&runtime));
        let deadline = tokio::time::Instant::now() + timeout.unwrap_or(DEFAULT_LIFECYCLE_TIMEOUT);
        self.pending_lifecycle = Some(PendingLifecycle::Shutdown {
            request_id,
            runtime: runtime.clone(),
            deadline,
        });
        if let Err(error) = self.write(&DriverMessage::Shutdown { request_id }).await {
            return self.retirement_after_error(error).await;
        }
        self.finish_shutdown(request_id, runtime, deadline).await
    }

    async fn finish_shutdown(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        deadline: tokio::time::Instant,
    ) -> IsolatedMexRetirement<MexWireError> {
        self.output.select_origin(Some(&runtime));
        let outcome = match self
            .wait_for_lifecycle_until(request_id, runtime, deadline, "shutdown")
            .await
        {
            Ok(outcome) => outcome,
            Err(error) => return self.retirement_after_error(error).await,
        };
        let diagnostic = match outcome {
            MexLifecycleOutcome::Retained => {
                self.pending_lifecycle = None;
                return IsolatedMexRetirement::Retained(wire_error(
                    "RunMat:MEX:Lifecycle",
                    "MEX host retained an active module during shutdown",
                    Some(self.module.display().to_string()),
                ));
            }
            MexLifecycleOutcome::Retired { diagnostic } => diagnostic,
            MexLifecycleOutcome::Cleared => {
                return IsolatedMexRetirement::Retained(wire_error(
                    "RunMat:MEX:HostProtocol",
                    "host returned a clear result for shutdown",
                    None,
                ));
            }
        };
        self.pending_lifecycle = Some(PendingLifecycle::ShutdownExit {
            diagnostic: diagnostic.clone(),
            deadline,
        });
        self.finish_shutdown_exit(diagnostic, deadline).await
    }

    async fn finish_shutdown_exit(
        &mut self,
        diagnostic: Option<MexWireError>,
        deadline: tokio::time::Instant,
    ) -> IsolatedMexRetirement<MexWireError> {
        let exit = match tokio::time::timeout_at(deadline, self.child.wait_and_retire_tree()).await
        {
            Err(_) => {
                let error = wire_error(
                    "RunMat:MEX:Timeout",
                    "isolated MEX shutdown did not finish before its deadline",
                    Some(self.module.display().to_string()),
                );
                if self.child.terminate_tree().await.is_ok() {
                    self.tree_retired = true;
                    self.pending_lifecycle = None;
                    return IsolatedMexRetirement::Retired(Err(error));
                }
                return IsolatedMexRetirement::Retained(error);
            }
            Ok(exit) => match exit {
                Ok(exit) => exit,
                Err(error) => {
                    return IsolatedMexRetirement::Retained(wire_error(
                        "RunMat:MEX:HostExit",
                        error.to_string(),
                        None,
                    ));
                }
            },
        };
        self.tree_retired = true;
        self.pending_lifecycle = None;
        if !exit.success {
            return IsolatedMexRetirement::Retired(Err(self.crash_error(exit.code)));
        }
        IsolatedMexRetirement::Retired(diagnostic.map_or(Ok(()), Err))
    }

    async fn wait_for_lifecycle_until(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        deadline: tokio::time::Instant,
        operation: &str,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        let outcome =
            tokio::time::timeout_at(deadline, self.wait_for_lifecycle(request_id, runtime)).await;
        match outcome {
            Ok(Ok(outcome)) => Ok(outcome),
            Ok(Err(error)) => {
                if matches!(
                    error.identifier.as_str(),
                    "RunMat:MEX:Cancelled" | "RunMat:MEX:HostCrashed"
                ) {
                    let _ = self.child.terminate_tree().await;
                }
                Err(error)
            }
            Err(_) => {
                let _ = self.child.terminate_tree().await;
                Err(wire_error(
                    "RunMat:MEX:Timeout",
                    format!("isolated MEX {operation} did not finish before its deadline"),
                    Some(self.module.display().to_string()),
                ))
            }
        }
    }

    async fn wait_for_lifecycle(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        loop {
            self.flush_pending_reply().await?;
            // Once a lifecycle command may have been written, request-scoped
            // cancellation cannot revoke it. The owning host must settle its
            // exact reply or observe process-tree retirement.
            let have_callbacks = self.callbacks.has_pending();
            let event = tokio::select! {
                message = self.messages.next() => ClientEvent::Message(message.unwrap_or_else(|| Err(wire_error("RunMat:MEX:HostTransport", "host message channel closed", None)))),
                callback = self.callbacks.next(), if have_callbacks => ClientEvent::Callback(callback.expect("active callback task")),
                _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
            };
            let message = match event {
                ClientEvent::Callback(response) => {
                    self.callbacks.finish(&response)?;
                    self.publish_callback_result(response).await?;
                    continue;
                }
                ClientEvent::Message(message) => message,
            };
            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.transport_or_crash(error).await),
            };
            match message {
                HostMessage::Lifecycle(result) if result.request_id == request_id => {
                    return result.outcome;
                }
                HostMessage::Callback(callback) => {
                    let origin = self
                        .origins
                        .get(&callback.request_id)
                        .cloned()
                        .unwrap_or_else(|| runtime.clone());
                    self.callbacks
                        .start(callback, origin, Rc::clone(&self.snapshots))?;
                }
                HostMessage::CallbackCancelled(cancelled) => {
                    self.callbacks.cancel(cancelled)?;
                }
                HostMessage::OriginReleased { request_id } => {
                    self.origins.remove(&request_id);
                    self.callbacks.release_request(request_id);
                }
                _ => {
                    return Err(wire_error(
                        "RunMat:MEX:HostProtocol",
                        "host record does not match the active lifecycle request",
                        None,
                    ));
                }
            }
        }
    }

    async fn write(&mut self, message: &DriverMessage) -> Result<(), MexWireError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        self.flush_pending_reply().await?;
        write_message(&mut self.writer, message, self.limits).await
    }

    fn poll_pending_reply(&mut self, context: &mut Context<'_>) -> Poll<Result<(), MexWireError>> {
        let Some(reply) = self.pending_reply.as_mut() else {
            return Poll::Ready(Ok(()));
        };
        match reply.poll_write(&mut self.writer, context) {
            Poll::Ready(Ok(())) => {
                self.pending_reply = None;
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(transport_error(error))),
            Poll::Pending => Poll::Pending,
        }
    }

    async fn flush_pending_reply(&mut self) -> Result<(), MexWireError> {
        let result = std::future::poll_fn(|context| self.poll_pending_reply(context)).await;
        if let Err(error) = &result {
            let _ = self.poison_transport(error.clone()).await;
        }
        result
    }

    /// Keep the original wire failure as public terminal authority, while
    /// exposing the exact independent retirement outcome to its owner/tests.
    /// Failed retirement leaves every pending owner intact for shutdown retry.
    async fn poison_transport(
        &mut self,
        error: MexWireError,
    ) -> runmat_process_host::ProcessHostResult<()> {
        if self.terminal_error.is_none() {
            self.terminal_error = Some(error);
        }
        self.retire_poisoned_host().await
    }

    async fn retire_poisoned_host(&mut self) -> runmat_process_host::ProcessHostResult<()> {
        if !self.tree_retired {
            self.child.terminate_tree().await?;
            self.tree_retired = true;
        }
        self.pending_invocation = None;
        self.pending_lifecycle = None;
        self.origins.clear();
        self.callbacks = ClientCallbackTasks::default();
        self.pending_reply = None;
        Ok(())
    }

    async fn publish_callback_result(
        &mut self,
        response: MexCallbackResult,
    ) -> Result<(), MexWireError> {
        self.flush_pending_reply().await?;
        let frame = match callback_reply_frame(response, self.limits) {
            Ok(frame) => frame,
            Err(error) => {
                let _ = self.poison_transport(error.clone()).await;
                return Err(error);
            }
        };
        self.pending_reply = Some(frame);
        self.flush_pending_reply().await
    }

    fn poll_background_reply(&mut self) -> Poll<Result<(), MexWireError>> {
        let waker = Waker::from(Arc::new(BackgroundReady(Arc::clone(
            &self.background_ready,
        ))));
        let mut context = Context::from_waker(&waker);
        self.poll_pending_reply(&mut context)
    }

    fn poll_ready_background_event(&mut self) -> Option<ClientEvent> {
        let ready = Arc::clone(&self.background_ready);
        let have_callbacks = self.callbacks.has_pending();
        match poll_background_once(
            async {
                tokio::select! {
                    message = self.messages.next() => ClientEvent::Message(message.unwrap_or_else(|| Err(wire_error("RunMat:MEX:HostTransport", "host message channel closed", None)))),
                    callback = self.callbacks.next(), if have_callbacks => ClientEvent::Callback(callback.expect("active callback task")),
                }
            },
            &ready,
        ) {
            Poll::Ready(event) => Some(event),
            Poll::Pending => None,
        }
    }

    async fn service_ready_background_protocol(&mut self) -> Result<bool, MexWireError> {
        if self.terminal_error.is_some() {
            return Ok(false);
        }
        if self.pending_reply.is_some() {
            return match self.poll_background_reply() {
                Poll::Ready(Err(error)) => {
                    let _ = self.poison_transport(error.clone()).await;
                    Err(error)
                }
                Poll::Ready(Ok(())) => Ok(true),
                Poll::Pending => Ok(false),
            };
        }
        if self.pending_lifecycle.is_some() {
            // Only the lifecycle owner may settle a pending clear/shutdown:
            // it must observe the typed outcome and (for shutdown) child exit.
            return Ok(false);
        }
        let Some(event) = self.poll_ready_background_event() else {
            return Ok(false);
        };
        let message = match event {
            ClientEvent::Callback(response) => {
                self.callbacks.finish(&response)?;
                let frame = match callback_reply_frame(response, self.limits) {
                    Ok(frame) => frame,
                    Err(error) => {
                        let _ = self.poison_transport(error.clone()).await;
                        return Err(error);
                    }
                };
                self.pending_reply = Some(frame);
                if let Poll::Ready(Err(error)) = self.poll_background_reply() {
                    let _ = self.poison_transport(error.clone()).await;
                    return Err(error);
                }
                return Ok(true);
            }
            ClientEvent::Message(message) => message,
        };
        let message = match message {
            Ok(message) => message,
            Err(error) if self.pending_invocation.is_some() => {
                return Err(self.transport_or_crash(error).await)
            }
            Err(error) => return Err(error),
        };
        match message {
            HostMessage::Invocation(result)
                if self.pending_invocation.as_ref().map(|(id, _)| *id)
                    == Some(result.request_id) =>
            {
                self.pending_invocation = None;
                let output = result.outcome?;
                // The abandoned batch still acquires roots before the host
                // transfer retires; no raw managed value is republished.
                let _outputs = decode_value_transfers_rooted(&output.outputs, &self.snapshots)?;
                Ok(true)
            }
            HostMessage::Callback(callback) => {
                let origin = self
                    .origins
                    .get(&callback.request_id)
                    .cloned()
                    .ok_or_else(|| {
                        wire_error(
                            "RunMat:MEX:CallbackProtocol",
                            "callback origin is no longer active",
                            None,
                        )
                    })?;
                self.callbacks
                    .start(callback, origin, Rc::clone(&self.snapshots))?;
                Ok(true)
            }
            HostMessage::CallbackCancelled(cancelled) => {
                self.callbacks.cancel(cancelled)?;
                Ok(true)
            }
            HostMessage::OriginReleased { request_id } => {
                self.origins.remove(&request_id);
                self.callbacks.release_request(request_id);
                Ok(true)
            }
            HostMessage::Invocation(_) | HostMessage::Lifecycle(_) => Err(wire_error(
                "RunMat:MEX:HostProtocol",
                "foreground host result arrived without an active request",
                None,
            )),
        }
    }

    fn take_request_id(&mut self) -> Result<u64, MexWireError> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.checked_add(1).ok_or_else(|| {
            wire_error(
                "RunMat:MEX:HostProtocol",
                "request identity exhausted",
                None,
            )
        })?;
        Ok(request_id)
    }

    fn crash_error(&self, code: Option<i32>) -> MexWireError {
        let detail = self.child.captured_stderr().text();
        wire_error(
            "RunMat:MEX:HostCrashed",
            format!(
                "isolated MEX host exited with code {code:?}{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            ),
            Some(self.module.display().to_string()),
        )
    }

    async fn retirement_after_error(
        &mut self,
        error: MexWireError,
    ) -> IsolatedMexRetirement<MexWireError> {
        if matches!(self.child.try_wait_unreaped(), Ok(Some(_)))
            && self.retire_poisoned_host().await.is_ok()
        {
            IsolatedMexRetirement::Retired(Err(error))
        } else {
            IsolatedMexRetirement::Retained(error)
        }
    }

    async fn transport_or_crash(&mut self, transport: MexWireError) -> MexWireError {
        for _ in 0..10 {
            match self.child.try_wait() {
                Ok(Some(exit)) => return self.crash_error(exit.code),
                Ok(None) => tokio::time::sleep(Duration::from_millis(10)).await,
                Err(_) => return transport,
            }
        }
        transport
    }
}

impl Drop for IsolatedMexClient {
    fn drop(&mut self) {
        let _ = self.child.try_wait();
    }
}

async fn handle_callback(
    callback: MexCallbackRequest,
    runtime: RuntimeContext,
    snapshots: Rc<SharedSnapshotStore>,
) -> MexCallbackResult {
    let request_id = callback.request_id;
    let callback_id = callback.callback_id;
    let outcome = match callback.validate() {
        Err(error) => Err(wire_error(
            "RunMat:MEX:CallbackProtocol",
            error.to_string(),
            None,
        )),
        Ok(()) => {
            let selected = runtime.clone();
            let admitted = crate::context::run_selected_callback_continuation_with(
                &runtime,
                move || async move {
                    Ok::<_, RuntimeError>(
                        poll_in_callback_scope(execute_callback(
                            callback.operation,
                            selected,
                            snapshots.as_ref(),
                        ))
                        .await,
                    )
                },
            )
            .await;
            match admitted {
                Ok(outcome) => outcome,
                Err(error) => Err(diagnostic_error(
                    super::super::services::runtime_diagnostic(error),
                )),
            }
        }
    };
    MexCallbackResult {
        request_id,
        callback_id,
        outcome,
    }
}

async fn poll_in_callback_scope<F>(future: F) -> F::Output
where
    F: Future,
{
    futures::pin_mut!(future);
    std::future::poll_fn(|context| {
        let _callback_scope = CallbackScope::enter();
        future.as_mut().poll(context)
    })
    .await
}

async fn execute_callback(
    operation: MexCallbackOperation,
    runtime: RuntimeContext,
    snapshots: &SharedSnapshotStore,
) -> Result<MexCallbackOutput, MexWireError> {
    let services = RuntimeMexHostServices::new(runtime);
    match operation {
        MexCallbackOperation::Eval { command } => services
            .eval_rooted(command)
            .await
            .map(|()| MexCallbackOutput::Unit)
            .map_err(|error| diagnostic_error(super::super::services::runtime_diagnostic(error))),
        MexCallbackOperation::Call {
            function,
            arguments,
            requested_outputs,
        } => {
            let arguments = decode_value_transfers_rooted(&arguments, snapshots)?;
            let values = arguments
                .with_unbounded_values_async(|arguments| {
                    services.call_rooted(&function, arguments, requested_outputs as usize)
                })
                .await
                .map_err(|error| {
                    diagnostic_error(super::super::services::runtime_diagnostic(error))
                })?
                .map_err(|error| {
                    diagnostic_error(super::super::services::runtime_diagnostic(error))
                })?
                .try_map_owned(
                    |sequence| {
                        let values = crate::sequence::ResolveValueSequence::resolve(
                            sequence,
                            runmat_types::SequenceUse::SelectPrefix {
                                count: requested_outputs as usize,
                            },
                            crate::sequence::SequenceResolutionContext::default(),
                        )?;
                        runmat_value::ValueSequence::from_output_storage(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
                .map_err(|error| {
                    diagnostic_error(super::super::services::runtime_diagnostic(error))
                })?;
            let mut transfers = Vec::with_capacity(values.len());
            for value in values.iter() {
                let host_value = crate::gather_if_needed_async(value)
                    .await
                    .map_err(|error| {
                        wire_error("RunMat:MEX:ValueGather", error.to_string(), None)
                    })?;
                transfers.push(encode_value_transfer(&host_value, snapshots)?);
            }
            Ok(MexCallbackOutput::Values(transfers))
        }
        MexCallbackOperation::GetVariable { workspace, name } => {
            let value = services
                .get_variable(&workspace, &name)
                .map_err(diagnostic_error)?;
            let transfer = if let Some(value) = value {
                let value = root_published_value(value)?;
                let host_value = crate::gather_if_needed_async(&value)
                    .await
                    .map_err(|error| {
                        wire_error("RunMat:MEX:ValueGather", error.to_string(), None)
                    })?;
                Some(encode_value_transfer(&host_value, snapshots)?)
            } else {
                None
            };
            Ok(MexCallbackOutput::OptionalValue(transfer))
        }
        MexCallbackOperation::PutVariable {
            workspace,
            name,
            value,
        } => {
            let value = decode_value_transfer_rooted_value(&value, snapshots)?;
            value
                .with_owned(|value| services.put_variable(&workspace, &name, value))
                .map(|()| MexCallbackOutput::Unit)
                .map_err(diagnostic_error)
        }
        MexCallbackOperation::GetObjectProperty {
            object,
            index,
            name,
        } => {
            let object = decode_value_transfer_rooted_value(&object, snapshots)?;
            let index = usize::try_from(index).map_err(|_| {
                wire_error(
                    "RunMat:MEX:ObjectIndex",
                    "object index exceeds the current platform width",
                    None,
                )
            })?;
            let value = object
                .with_owned_async(|object| services.get_object_property_rooted(object, index, name))
                .await
                .map_err(diagnostic_error)?;
            let host_value = crate::gather_if_needed_async(&value)
                .await
                .map_err(|error| wire_error("RunMat:MEX:ValueGather", error.to_string(), None))?;
            Ok(MexCallbackOutput::Values(vec![encode_value_transfer(
                &host_value,
                snapshots,
            )?]))
        }
        MexCallbackOperation::SetObjectProperty {
            object,
            index,
            name,
            value,
        } => {
            let object = decode_value_transfer_rooted_value(&object, snapshots)?;
            let value = decode_value_transfer_rooted_value(&value, snapshots)?;
            let index = usize::try_from(index).map_err(|_| {
                wire_error(
                    "RunMat:MEX:ObjectIndex",
                    "object index exceeds the current platform width",
                    None,
                )
            })?;
            let object = object
                .with_owned_async(|object| {
                    value.with_owned_async(|value| {
                        services.set_object_property_rooted(object, index, name, value)
                    })
                })
                .await
                .map_err(diagnostic_error)?;
            let host_value = crate::gather_if_needed_async(&object)
                .await
                .map_err(|error| wire_error("RunMat:MEX:ValueGather", error.to_string(), None))?;
            Ok(MexCallbackOutput::Values(vec![encode_value_transfer(
                &host_value,
                snapshots,
            )?]))
        }
    }
}

async fn read_message(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> Result<HostMessage, MexWireError> {
    let payload = read_payload(reader, limits)
        .await
        .map_err(transport_error)?;
    let message: HostMessage = serde_json::from_slice(&payload)
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    message
        .validate()
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    Ok(message)
}

async fn write_message(
    writer: &mut (impl AsyncWrite + Unpin),
    message: &DriverMessage,
    limits: FrameLimits,
) -> Result<(), MexWireError> {
    message
        .validate()
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    let payload = serde_json::to_vec(message)
        .map_err(|error| wire_error("RunMat:MEX:HostProtocol", error.to_string(), None))?;
    write_payload(writer, &payload, limits)
        .await
        .map_err(transport_error)
}

fn transport_error(error: runmat_process_host::ProcessHostError) -> MexWireError {
    wire_error("RunMat:MEX:HostTransport", error.to_string(), None)
}

fn diagnostic_error(error: MexDiagnostic) -> MexWireError {
    wire_error(
        error.identifier.as_deref().unwrap_or("RunMat:MEX:Callback"),
        error.message,
        None,
    )
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use runmat_process_host::is_process_alive;
    use tokio::io::AsyncReadExt;

    async fn sleeping_test_client(
        module: &str,
        messages: async_mpsc::UnboundedReceiver<Result<HostMessage, MexWireError>>,
        pending_invocation: Option<(u64, RuntimeContext)>,
    ) -> IsolatedMexClient {
        let mut command = HostCommand::new("/bin/sleep");
        command.arguments = vec!["2".into()];
        let mut child = command.spawn().await.unwrap();
        let stdio = child.take_stdio().unwrap();
        IsolatedMexClient {
            module: PathBuf::from(module),
            child,
            snapshots: Rc::new(SharedSnapshotStore::create().unwrap()),
            messages,
            writer: FramedWriter::Fixture(stdio.stdin),
            output: HostOutput::default(),
            limits: FrameLimits {
                max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
            },
            next_request_id: 3,
            origins: HashMap::new(),
            callbacks: ClientCallbackTasks::default(),
            pending_reply: None,
            terminal_error: None,
            tree_retired: false,
            background_ready: Arc::new(tokio::sync::Notify::new()),
            pending_invocation,
            pending_lifecycle: None,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pending_first_module_does_not_starve_ready_second_module() {
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let (first_sender, first_messages) = async_mpsc::unbounded();
        let (second_sender, second_messages) = async_mpsc::unbounded();
        let mut first = sleeping_test_client("first.mex", first_messages, Some((1, runtime))).await;
        let mut second = sleeping_test_client("second.mex", second_messages, None).await;
        second_sender
            .unbounded_send(Ok(HostMessage::OriginReleased { request_id: 2 }))
            .unwrap();

        assert!(!first.service_ready_background().await.unwrap());
        assert!(second.service_ready_background().await.unwrap());
        assert!(first.pending_invocation.is_some());
        drop(first_sender);
        first.child.terminate_tree().await.unwrap();
        second.child.terminate_tree().await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn poisoned_reply_transport_retires_host_and_blocks_second_admission() {
        let (sender, messages) = async_mpsc::unbounded();
        let mut client = sleeping_test_client("poisoned.mex", messages, None).await;
        let original = wire_error(
            "RunMat:MEX:HostTransport",
            "partial callback frame failed",
            None,
        );
        client
            .poison_transport(original.clone())
            .await
            .expect("poisoned host must prove process-tree retirement");
        assert_eq!(client.terminal_error, Some(original.clone()));
        assert!(client.tree_retired);
        assert!(client.child.try_wait().unwrap().is_some());
        assert!(!client.service_ready_background().await.unwrap());
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
        let clear_error = client
            .clear(runtime.clone(), None)
            .await
            .expect_err("poisoned transport rejects clear");
        assert_eq!(
            std::error::Error::source(&clear_error)
                .and_then(|cause| cause.downcast_ref::<MexWireError>()),
            Some(&original)
        );
        assert!(matches!(
            client.shutdown(runtime, None).await,
            IsolatedMexRetirement::Retired(Err(error))
                if std::error::Error::source(&error)
                    .and_then(|cause| cause.downcast_ref::<MexWireError>()) == Some(&original)
        ));
        drop(sender);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exited_host_leader_still_retires_its_live_process_group() {
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec!["-c".into(), "sleep 60 & echo $!; exit 0".into()];
        let mut child = command.spawn().await.unwrap();
        let mut stdio = child.take_stdio().unwrap();
        let mut pid_line = Vec::new();
        loop {
            let mut byte = [0];
            stdio.stdout.read_exact(&mut byte).await.unwrap();
            if byte[0] == b'\n' {
                break;
            }
            pid_line.push(byte[0]);
        }
        let descendant: u32 = String::from_utf8(pid_line).unwrap().parse().unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while child.try_wait().unwrap().is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shell leader exits before group retirement");
        assert!(is_process_alive(descendant));

        let (_sender, messages) = async_mpsc::unbounded();
        let mut client = IsolatedMexClient {
            module: PathBuf::from("exited-leader.mex"),
            child,
            snapshots: Rc::new(SharedSnapshotStore::create().unwrap()),
            messages,
            writer: FramedWriter::Fixture(stdio.stdin),
            output: HostOutput::default(),
            limits: FrameLimits {
                max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
            },
            next_request_id: 1,
            origins: HashMap::new(),
            callbacks: ClientCallbackTasks::default(),
            pending_reply: None,
            terminal_error: None,
            tree_retired: false,
            background_ready: Arc::new(tokio::sync::Notify::new()),
            pending_invocation: None,
            pending_lifecycle: None,
        };
        client
            .poison_transport(wire_error(
                "RunMat:MEX:HostTransport",
                "leader already exited",
                None,
            ))
            .await
            .expect("exited host leader must still prove process-group retirement");
        assert!(client.tree_retired);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while is_process_alive(descendant) {
            assert!(
                tokio::time::Instant::now() < deadline,
                "descendant survived confirmed tree retirement"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retired_reply_without_child_exit_uses_same_shutdown_deadline() {
        let mut command = HostCommand::new("/bin/sleep");
        command.arguments = vec!["2".into()];
        let mut child = command.spawn().await.unwrap();
        let stdio = child.take_stdio().unwrap();
        let (sender, messages) = async_mpsc::unbounded();
        sender
            .unbounded_send(Ok(HostMessage::Lifecycle(
                super::super::MexLifecycleResult {
                    request_id: 1,
                    outcome: Ok(MexLifecycleOutcome::Retired { diagnostic: None }),
                },
            )))
            .unwrap();
        let runtime = RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let mut client = IsolatedMexClient {
            module: PathBuf::from("stalled.mex"),
            child,
            snapshots: Rc::new(SharedSnapshotStore::create().unwrap()),
            messages,
            writer: FramedWriter::Fixture(stdio.stdin),
            output: HostOutput::default(),
            limits: FrameLimits {
                max_message_bytes: MEX_HOST_MAX_MESSAGE_BYTES,
            },
            next_request_id: 2,
            origins: HashMap::new(),
            callbacks: ClientCallbackTasks::default(),
            pending_reply: None,
            terminal_error: None,
            tree_retired: false,
            background_ready: Arc::new(tokio::sync::Notify::new()),
            pending_invocation: None,
            pending_lifecycle: Some(PendingLifecycle::Shutdown {
                request_id: 1,
                runtime: runtime.clone(),
                deadline: tokio::time::Instant::now() + Duration::from_millis(50),
            }),
        };

        let retirement = client.shutdown(runtime, Some(Duration::from_secs(3))).await;
        assert!(matches!(
            retirement,
            IsolatedMexRetirement::Retired(Err(error))
                if error.identifier() == Some("RunMat:MEX:Timeout")
        ));
        assert!(client.child.try_wait().unwrap().is_some());
    }
}
