use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_process_host::{ChildProcess, HiddenMode, HostCommand};
use runmat_value::{Value, WeakForeignRef};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{
    decode_portable, encode_portable, wire_error, NativeCallbackRequest, NativeCallbackResult,
    NativeDriverMessage, NativeHostMessage, NativeInvocationRequest, NativeRemoteReference,
    NativeShutdownResult, NativeWireError, NativeWireValue, NATIVE_FFI_HOST_KIND,
    NATIVE_FFI_HOST_KIND_ENV, NATIVE_FFI_HOST_MAX_MESSAGE_BYTES, NATIVE_FFI_HOST_PROTOCOL,
    NATIVE_FFI_HOST_SCHEMA_VERSION, NATIVE_FFI_HOST_SECRET_ENV, NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{ForeignCall, RuntimeContext};
use crate::foreign::{ForeignErrorKind, ForeignHandleRegistry, ForeignResourceMetadata};
use crate::{build_runtime_error, foreign::foreign_error, RuntimeError};

const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeFfiIsolationPolicy {
    pub isolate: bool,
    pub invocation_timeout: Option<Duration>,
}

impl NativeFfiIsolationPolicy {
    pub const fn in_process() -> Self {
        Self {
            isolate: false,
            invocation_timeout: None,
        }
    }

    pub const fn isolated(invocation_timeout: Option<Duration>) -> Self {
        Self {
            isolate: true,
            invocation_timeout,
        }
    }
}

impl Default for NativeFfiIsolationPolicy {
    fn default() -> Self {
        Self::in_process()
    }
}

pub struct IsolatedNativeFfiClient {
    child: ChildProcess,
    snapshots: SharedSnapshotStore,
    reader: tokio::process::ChildStdout,
    writer: tokio::process::ChildStdin,
    limits: FrameLimits,
    next_request_id: u64,
    next_callback_id: u64,
    local_to_remote: BTreeMap<u64, u64>,
    remote_to_local: BTreeMap<u64, (u64, WeakForeignRef)>,
}

impl IsolatedNativeFfiClient {
    pub async fn spawn() -> Result<Self, NativeWireError> {
        let executable = std::env::current_exe()
            .map_err(|error| wire_error("RunMat:NativeFFI:HostExecutable", error.to_string()))?;
        let secret = SessionSecret::generate();
        let snapshots = SharedSnapshotStore::create()
            .map_err(|error| wire_error("RunMat:NativeFFI:ValueSnapshot", error.to_string()))?;
        let mut command = HostCommand::new(executable);
        command.arguments = vec![HiddenMode::ExtensionHost.marker().into()];
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        command
            .environment
            .insert(NATIVE_FFI_HOST_KIND_ENV.into(), NATIVE_FFI_HOST_KIND.into());
        command
            .environment
            .insert(NATIVE_FFI_HOST_SECRET_ENV.into(), secret.expose_hex());
        let snapshot_root = snapshots.root_path().to_str().ok_or_else(|| {
            wire_error(
                "RunMat:NativeFFI:ValueSnapshot",
                "shared snapshot root is not valid Unicode",
            )
        })?;
        command.environment.insert(
            NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV.into(),
            snapshot_root.into(),
        );
        let mut child = command
            .spawn()
            .await
            .map_err(|error| wire_error("RunMat:NativeFFI:HostSpawn", error.to_string()))?;
        let stdio = child
            .take_stdio()
            .map_err(|error| wire_error("RunMat:NativeFFI:HostTransport", error.to_string()))?;
        let mut reader = stdio.stdout;
        let mut writer = stdio.stdin;
        let session = authenticate_driver(
            &mut reader,
            &mut writer,
            HostHandshake::new(
                NATIVE_FFI_HOST_PROTOCOL,
                NATIVE_FFI_HOST_SCHEMA_VERSION,
                NATIVE_FFI_HOST_MAX_MESSAGE_BYTES,
            ),
            &secret,
        )
        .await
        .map_err(|error| wire_error("RunMat:NativeFFI:HostAuthentication", error.to_string()))?;
        Ok(Self {
            child,
            snapshots,
            reader,
            writer,
            limits: session.limits,
            next_request_id: 1,
            next_callback_id: 1,
            local_to_remote: BTreeMap::new(),
            remote_to_local: BTreeMap::new(),
        })
    }

    pub async fn invoke(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        released_local_handles: Vec<u64>,
        timeout: Option<Duration>,
    ) -> Result<Value, RuntimeError> {
        let request_id = self.take_request_id().map_err(runtime_error)?;
        let releases = self.translate_releases(released_local_handles);
        let mut callbacks = BTreeMap::new();
        let mut arguments = Vec::with_capacity(call.arguments.len());
        for value in &call.arguments {
            arguments.push(
                self.encode_driver_value(value, &mut callbacks)
                    .await
                    .map_err(runtime_error)?,
            );
        }
        let request = NativeDriverMessage::Invoke(NativeInvocationRequest {
            request_id,
            operation: call.symbol,
            arguments,
            requested_outputs: u32::try_from(call.requested_outputs).map_err(|_| {
                foreign_error(
                    ForeignErrorKind::InvalidCall,
                    "native FFI output count exceeds u32",
                )
            })?,
            releases,
        });
        self.write(&request).await.map_err(runtime_error)?;
        let wait = self.wait_for_invocation(
            request_id,
            runtime.clone(),
            callbacks,
            handles,
            host_identity,
        );
        let outcome = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, wait).await {
                Ok(outcome) => outcome,
                Err(_) => {
                    let _ = self.child.terminate_tree().await;
                    return Err(runtime_error(wire_error(
                        "RunMat:NativeFFI:Timeout",
                        format!(
                            "isolated native-library invocation exceeded {} ms",
                            timeout.as_millis()
                        ),
                    )));
                }
            }
        } else {
            wait.await
        };
        if outcome.as_ref().is_err_and(|error| {
            matches!(
                error.identifier(),
                Some(
                    "RunMat:NativeFFI:Timeout"
                        | "RunMat:NativeFFI:Cancelled"
                        | "RunMat:NativeFFI:HostCrashed"
                )
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
        callbacks: BTreeMap<u64, Value>,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<Value, RuntimeError> {
        loop {
            if runtime.cancellation().load(Ordering::Relaxed) {
                return Err(runtime_error(wire_error(
                    "RunMat:NativeFFI:Cancelled",
                    "isolated native-library invocation was cancelled",
                )));
            }
            let message = tokio::select! {
                message = read_message(&mut self.reader, self.limits) => message,
                _ = tokio::time::sleep(Duration::from_millis(25)) => continue,
            };
            let message = match message {
                Ok(message) => message,
                Err(error) => return Err(runtime_error(self.transport_or_crash(error).await)),
            };
            match message {
                NativeHostMessage::Invocation(result) if result.request_id == request_id => {
                    let value = result.outcome.map_err(runtime_error)?;
                    return self
                        .decode_driver_value(value, handles, host_identity)
                        .map_err(runtime_error);
                }
                NativeHostMessage::Callback(callback) if callback.request_id == request_id => {
                    let response = self
                        .handle_callback(callback, runtime.clone(), &callbacks)
                        .await;
                    self.write(&NativeDriverMessage::CallbackResult(response))
                        .await
                        .map_err(runtime_error)?;
                }
                _ => {
                    return Err(runtime_error(wire_error(
                        "RunMat:NativeFFI:HostProtocol",
                        "host record does not match the active native-library invocation",
                    )))
                }
            }
        }
    }

    async fn handle_callback(
        &mut self,
        callback: NativeCallbackRequest,
        runtime: RuntimeContext,
        callbacks: &BTreeMap<u64, Value>,
    ) -> NativeCallbackResult {
        let outcome = async {
            let callable = callbacks
                .get(&callback.callback_id)
                .cloned()
                .ok_or_else(|| {
                    wire_error(
                        "RunMat:NativeFFI:CallbackProtocol",
                        "host requested an unknown native callback",
                    )
                })?;
            let mut arguments = Vec::with_capacity(callback.arguments.len());
            for value in callback.arguments {
                match value {
                    NativeWireValue::Portable(transfer) => {
                        arguments.push(decode_portable(&transfer, &self.snapshots)?);
                    }
                    _ => {
                        return Err(wire_error(
                            "RunMat:NativeFFI:CallbackProtocol",
                            "native callback arguments must be portable values",
                        ))
                    }
                }
            }
            let result = runtime
                .scope(crate::call_feval_async_with_outputs(
                    callable, &arguments, 1,
                ))
                .await
                .map_err(|error| {
                    wire_error("RunMat:NativeFFI:CallbackFailed", error.to_string())
                })?;
            let host_value = crate::gather_if_needed_async(&result)
                .await
                .map_err(|error| wire_error("RunMat:NativeFFI:ValueGather", error.to_string()))?;
            encode_portable(&host_value, &self.snapshots).map(NativeWireValue::Portable)
        }
        .await;
        NativeCallbackResult {
            request_id: callback.request_id,
            callback_id: callback.callback_id,
            outcome,
        }
    }

    async fn encode_driver_value(
        &mut self,
        value: &Value,
        callbacks: &mut BTreeMap<u64, Value>,
    ) -> Result<NativeWireValue, NativeWireError> {
        if super::super::is_callable(value) {
            let id = self.next_callback_id;
            self.next_callback_id = self.next_callback_id.checked_add(1).ok_or_else(|| {
                wire_error(
                    "RunMat:NativeFFI:HostProtocol",
                    "callback identity exhausted",
                )
            })?;
            callbacks.insert(id, value.clone());
            return Ok(NativeWireValue::Callback { id });
        }
        match value {
            Value::Foreign(reference) => {
                let id = self
                    .local_to_remote
                    .get(&reference.handle)
                    .copied()
                    .ok_or_else(|| {
                        wire_error(
                            "RunMat:NativeFFI:StaleHandle",
                            "native pointer does not belong to the active isolated host",
                        )
                    })?;
                Ok(NativeWireValue::Foreign(NativeRemoteReference {
                    id,
                    type_identity: reference.type_identity.clone(),
                    ownership: reference.ownership,
                    affinity: reference.affinity,
                    lifetime: reference.lifetime,
                }))
            }
            Value::OutputList(values) => {
                let mut encoded = Vec::with_capacity(values.len());
                for value in values {
                    encoded.push(Box::pin(self.encode_driver_value(value, callbacks)).await?);
                }
                Ok(NativeWireValue::OutputList(encoded))
            }
            _ => {
                let host_value = crate::gather_if_needed_async(value)
                    .await
                    .map_err(|error| {
                        wire_error("RunMat:NativeFFI:ValueGather", error.to_string())
                    })?;
                encode_portable(&host_value, &self.snapshots).map(NativeWireValue::Portable)
            }
        }
    }

    fn decode_driver_value(
        &mut self,
        value: NativeWireValue,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<Value, NativeWireError> {
        match value {
            NativeWireValue::Portable(transfer) => decode_portable(&transfer, &self.snapshots),
            NativeWireValue::Foreign(remote) => {
                let reference = if let Some((_, reference)) = self.remote_to_local.get(&remote.id) {
                    reference.upgrade().ok_or_else(|| {
                        wire_error(
                            "RunMat:NativeFFI:StaleHandle",
                            "native pointer was released before the isolated host returned it",
                        )
                    })?
                } else {
                    let reference = handles
                        .register_resource(
                            host_identity,
                            ForeignResourceMetadata {
                                type_identity: remote.type_identity,
                                ownership: remote.ownership,
                                affinity: remote.affinity,
                                lifetime: remote.lifetime,
                            },
                        )
                        .map_err(|error| {
                            wire_error("RunMat:NativeFFI:HostUnavailable", error.to_string())
                        })?;
                    self.local_to_remote.insert(reference.handle, remote.id);
                    let weak = reference.downgrade().ok_or_else(|| {
                        wire_error(
                            "RunMat:NativeFFI:HostProtocol",
                            "isolated native pointer is missing its managed lease",
                        )
                    })?;
                    self.remote_to_local
                        .insert(remote.id, (reference.handle, weak));
                    reference
                };
                Ok(Value::Foreign(reference))
            }
            NativeWireValue::OutputList(values) => values
                .into_iter()
                .map(|value| self.decode_driver_value(value, handles, host_identity))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::OutputList),
            NativeWireValue::Callback { .. } => Err(wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "host returned a callback token as a value",
            )),
        }
    }

    fn translate_releases(&mut self, local_handles: Vec<u64>) -> Vec<u64> {
        local_handles
            .into_iter()
            .filter_map(|local| {
                let remote = self.local_to_remote.remove(&local)?;
                self.remote_to_local.remove(&remote);
                Some(remote)
            })
            .collect()
    }

    pub async fn shutdown(
        &mut self,
        released_local_handles: Vec<u64>,
    ) -> Result<(), NativeWireError> {
        let request_id = self.take_request_id()?;
        let releases = self.translate_releases(released_local_handles);
        self.write(&NativeDriverMessage::Shutdown {
            request_id,
            releases,
        })
        .await?;
        let outcome = tokio::time::timeout(
            DEFAULT_SHUTDOWN_TIMEOUT,
            read_message(&mut self.reader, self.limits),
        )
        .await
        .map_err(|_| {
            wire_error(
                "RunMat:NativeFFI:Timeout",
                "isolated native-library shutdown timed out",
            )
        })??;
        let NativeHostMessage::Shutdown(NativeShutdownResult {
            request_id: response_id,
            outcome,
        }) = outcome
        else {
            return Err(wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "host returned the wrong shutdown record",
            ));
        };
        if response_id != request_id {
            return Err(wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "shutdown response identity does not match the request",
            ));
        }
        outcome?;
        let exit = self
            .child
            .wait()
            .await
            .map_err(|error| wire_error("RunMat:NativeFFI:HostExit", error.to_string()))?;
        if !exit.success {
            return Err(self.crash_error(exit.code));
        }
        Ok(())
    }

    async fn write(&mut self, message: &NativeDriverMessage) -> Result<(), NativeWireError> {
        write_message(&mut self.writer, message, self.limits).await
    }

    fn take_request_id(&mut self) -> Result<u64, NativeWireError> {
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.checked_add(1).ok_or_else(|| {
            wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "request identity exhausted",
            )
        })?;
        Ok(request_id)
    }

    fn crash_error(&self, code: Option<i32>) -> NativeWireError {
        let detail = self.child.captured_stderr().text();
        wire_error(
            "RunMat:NativeFFI:HostCrashed",
            format!(
                "isolated native-library host exited with code {code:?}{}",
                if detail.is_empty() {
                    String::new()
                } else {
                    format!(": {detail}")
                }
            ),
        )
    }

    async fn transport_or_crash(&mut self, transport: NativeWireError) -> NativeWireError {
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

impl Drop for IsolatedNativeFfiClient {
    fn drop(&mut self) {
        let _ = self.child.try_wait();
    }
}

async fn read_message(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> Result<NativeHostMessage, NativeWireError> {
    let payload = read_payload(reader, limits)
        .await
        .map_err(|error| wire_error("RunMat:NativeFFI:HostTransport", error.to_string()))?;
    let message: NativeHostMessage = serde_json::from_slice(&payload)
        .map_err(|error| wire_error("RunMat:NativeFFI:HostProtocol", error.to_string()))?;
    message
        .validate()
        .map_err(|error| wire_error("RunMat:NativeFFI:HostProtocol", error))?;
    Ok(message)
}

async fn write_message(
    writer: &mut (impl AsyncWrite + Unpin),
    message: &NativeDriverMessage,
    limits: FrameLimits,
) -> Result<(), NativeWireError> {
    message
        .validate()
        .map_err(|error| wire_error("RunMat:NativeFFI:HostProtocol", error))?;
    let payload = serde_json::to_vec(message)
        .map_err(|error| wire_error("RunMat:NativeFFI:HostProtocol", error.to_string()))?;
    write_payload(writer, &payload, limits)
        .await
        .map_err(|error| wire_error("RunMat:NativeFFI:HostTransport", error.to_string()))
}

fn runtime_error(error: NativeWireError) -> RuntimeError {
    build_runtime_error(error.message)
        .with_builtin("native_ffi")
        .with_identifier(error.identifier)
        .build()
}
