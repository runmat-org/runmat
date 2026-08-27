use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use runmat_mex::{MexApi, MexDiagnostic, MexHostServices, MexInvocation};
use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_process_host::{ChildProcess, HiddenMode, HostCommand};
use runmat_value::Value;
use tokio::io::{AsyncRead, AsyncWrite};

use super::{
    decode_value_transfer, encode_value_transfer, DriverMessage, HostMessage, MexBinaryTier,
    MexCallbackOperation, MexCallbackOutput, MexCallbackRequest, MexCallbackResult,
    MexInvocationRequest, MexLifecycleOutcome, MexWireError, MEX_HOST_MAX_MESSAGE_BYTES,
    MEX_HOST_PROTOCOL, MEX_HOST_SCHEMA_VERSION, MEX_HOST_SECRET_ENV, MEX_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::RuntimeContext;
use crate::foreign::RuntimeMexHostServices;

const DEFAULT_LIFECYCLE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct IsolatedMexClient {
    module: PathBuf,
    child: ChildProcess,
    snapshots: SharedSnapshotStore,
    reader: tokio::process::ChildStdout,
    writer: tokio::process::ChildStdin,
    limits: FrameLimits,
    next_request_id: u64,
}

impl IsolatedMexClient {
    pub async fn spawn(module: PathBuf) -> Result<Self, MexWireError> {
        let executable = std::env::current_exe()
            .map_err(|error| wire_error("RunMat:MEX:HostExecutable", error.to_string(), None))?;
        let secret = SessionSecret::generate();
        let snapshots = SharedSnapshotStore::create()
            .map_err(|error| wire_error("RunMat:MEX:ValueSnapshot", error.to_string(), None))?;
        let mut command = HostCommand::new(executable);
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
        let mut child = command
            .spawn()
            .await
            .map_err(|error| wire_error("RunMat:MEX:HostSpawn", error.to_string(), None))?;
        let stdio = child
            .take_stdio()
            .map_err(|error| wire_error("RunMat:MEX:HostTransport", error.to_string(), None))?;
        let mut reader = stdio.stdout;
        let mut writer = stdio.stdin;
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
        Ok(Self {
            module,
            child,
            snapshots,
            reader,
            writer,
            limits: session.limits,
            next_request_id: 1,
        })
    }

    pub async fn invoke(
        &mut self,
        arguments: &[Value],
        requested_outputs: usize,
        api: Option<MexApi>,
        tier: MexBinaryTier,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<MexInvocation, MexWireError> {
        let request_id = self.take_request_id()?;
        let mut transferred_arguments = Vec::with_capacity(arguments.len());
        for value in arguments {
            let host_value = crate::gather_if_needed_async(value)
                .await
                .map_err(|error| wire_error("RunMat:MEX:ValueGather", error.to_string(), None))?;
            transferred_arguments.push(encode_value_transfer(&host_value, &self.snapshots)?);
        }
        let request = DriverMessage::Invoke(MexInvocationRequest {
            request_id,
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
        self.write(&request).await?;
        let wait = self.wait_for_invocation(request_id, runtime.clone());
        let outcome = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, wait).await {
                Ok(outcome) => outcome,
                Err(_) => {
                    let _ = self.child.terminate_tree().await;
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
        }) {
            let _ = self.child.terminate_tree().await;
        }
        outcome
    }

    async fn wait_for_invocation(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
    ) -> Result<MexInvocation, MexWireError> {
        loop {
            if runtime.cancellation().load(Ordering::Relaxed) {
                return Err(wire_error(
                    "RunMat:MEX:Cancelled",
                    "isolated MEX invocation was cancelled",
                    Some(self.module.display().to_string()),
                ));
            }
            let message = tokio::select! {
                message = read_message(&mut self.reader, self.limits) => message,
                _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
            };
            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.transport_or_crash(error).await),
            };
            match message {
                HostMessage::Invocation(result) if result.request_id == request_id => {
                    let output = result.outcome?;
                    let outputs = output
                        .outputs
                        .iter()
                        .map(|value| decode_value_transfer(value, &self.snapshots))
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(MexInvocation {
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
                HostMessage::Callback(callback) if callback.request_id == request_id => {
                    let response =
                        handle_callback(callback, runtime.clone(), &self.snapshots).await;
                    self.write(&DriverMessage::CallbackResult(response)).await?;
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

    pub async fn clear(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        let request_id = self.take_request_id()?;
        self.write(&DriverMessage::Clear {
            request_id,
            module_path: self.module.display().to_string(),
        })
        .await?;
        self.wait_for_lifecycle_bounded(request_id, runtime, timeout, "clear")
            .await
    }

    pub async fn shutdown(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<(), MexWireError> {
        let request_id = self.take_request_id()?;
        self.write(&DriverMessage::Shutdown { request_id }).await?;
        let outcome = self
            .wait_for_lifecycle_bounded(request_id, runtime, timeout, "shutdown")
            .await?;
        if outcome != MexLifecycleOutcome::Shutdown {
            return Err(wire_error(
                "RunMat:MEX:HostProtocol",
                "host returned the wrong shutdown outcome",
                None,
            ));
        }
        let exit = self
            .child
            .wait()
            .await
            .map_err(|error| wire_error("RunMat:MEX:HostExit", error.to_string(), None))?;
        if !exit.success {
            return Err(self.crash_error(exit.code));
        }
        Ok(())
    }

    async fn wait_for_lifecycle_bounded(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
        operation: &str,
    ) -> Result<MexLifecycleOutcome, MexWireError> {
        let timeout = timeout.unwrap_or(DEFAULT_LIFECYCLE_TIMEOUT);
        let outcome =
            tokio::time::timeout(timeout, self.wait_for_lifecycle(request_id, runtime)).await;
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
                    format!(
                        "isolated MEX {operation} exceeded {} ms",
                        timeout.as_millis()
                    ),
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
            if runtime.cancellation().load(Ordering::Relaxed) {
                return Err(wire_error(
                    "RunMat:MEX:Cancelled",
                    "isolated MEX lifecycle operation was cancelled",
                    Some(self.module.display().to_string()),
                ));
            }
            let message = tokio::select! {
                message = read_message(&mut self.reader, self.limits) => message,
                _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
            };
            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(self.transport_or_crash(error).await),
            };
            match message {
                HostMessage::Lifecycle(result) if result.request_id == request_id => {
                    return result.outcome;
                }
                HostMessage::Callback(callback) if callback.request_id == request_id => {
                    let response =
                        handle_callback(callback, runtime.clone(), &self.snapshots).await;
                    self.write(&DriverMessage::CallbackResult(response)).await?;
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
        write_message(&mut self.writer, message, self.limits).await
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
    snapshots: &SharedSnapshotStore,
) -> MexCallbackResult {
    let request_id = callback.request_id;
    let callback_id = callback.callback_id;
    let outcome = match callback.validate() {
        Ok(()) => execute_callback(callback.operation, runtime, snapshots).await,
        Err(error) => Err(wire_error(
            "RunMat:MEX:CallbackProtocol",
            error.to_string(),
            None,
        )),
    };
    MexCallbackResult {
        request_id,
        callback_id,
        outcome,
    }
}

async fn execute_callback(
    operation: MexCallbackOperation,
    runtime: RuntimeContext,
    snapshots: &SharedSnapshotStore,
) -> Result<MexCallbackOutput, MexWireError> {
    let services = RuntimeMexHostServices::new(runtime);
    match operation {
        MexCallbackOperation::Eval { command } => services
            .eval(&command)
            .map(|()| MexCallbackOutput::Unit)
            .map_err(diagnostic_error),
        MexCallbackOperation::Call {
            function,
            arguments,
            requested_outputs,
        } => {
            let arguments = arguments
                .iter()
                .map(|value| decode_value_transfer(value, snapshots))
                .collect::<Result<Vec<_>, _>>()?;
            let values = services
                .call(&function, arguments, requested_outputs as usize)
                .map_err(diagnostic_error)?;
            let mut transfers = Vec::with_capacity(values.len());
            for value in &values {
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
            let value = decode_value_transfer(&value, snapshots)?;
            services
                .put_variable(&workspace, &name, value)
                .map(|()| MexCallbackOutput::Unit)
                .map_err(diagnostic_error)
        }
        MexCallbackOperation::GetObjectProperty {
            object,
            index,
            name,
        } => {
            let object = decode_value_transfer(&object, snapshots)?;
            let index = usize::try_from(index).map_err(|_| {
                wire_error(
                    "RunMat:MEX:ObjectIndex",
                    "object index exceeds the current platform width",
                    None,
                )
            })?;
            let value = services
                .get_object_property_at(object, index, &name)
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
            let object = decode_value_transfer(&object, snapshots)?;
            let value = decode_value_transfer(&value, snapshots)?;
            let index = usize::try_from(index).map_err(|_| {
                wire_error(
                    "RunMat:MEX:ObjectIndex",
                    "object index exceeds the current platform width",
                    None,
                )
            })?;
            let object = services
                .set_object_property_at(object, index, &name, value)
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
