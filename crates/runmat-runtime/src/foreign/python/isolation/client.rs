use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_process_host::{ChildProcess, HiddenMode, HostCommand};
use runmat_python::{discover_python, PythonDiscoveryRequest};
use runmat_value::{Value, WeakForeignRef};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::oneshot;

use super::{
    decode_portable, encode_portable, wire_error, PythonCallbackRequest, PythonCallbackResult,
    PythonDriverMessage, PythonHostMessage, PythonInvocationRequest, PythonRemoteReference,
    PythonShutdownResult, PythonWireError, PythonWireValue, PYTHON_HOST_CONFIG_ENV,
    PYTHON_HOST_KIND, PYTHON_HOST_KIND_ENV, PYTHON_HOST_MAX_MESSAGE_BYTES, PYTHON_HOST_PROTOCOL,
    PYTHON_HOST_SCHEMA_VERSION, PYTHON_HOST_SECRET_ENV, PYTHON_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{ForeignCall, RuntimeContext};
use crate::foreign::{
    foreign_callback_request, invoke_foreign_callback, ForeignHandleRegistry,
    ForeignResourceMetadata, PythonRuntimeConfiguration,
};
use crate::{build_runtime_error, RuntimeError};

const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

pub struct NestedPythonCall {
    pub call: ForeignCall,
    pub reply: oneshot::Sender<Result<Vec<Value>, RuntimeError>>,
}

pub type NestedPythonQueue = Rc<RefCell<VecDeque<NestedPythonCall>>>;

pub struct IsolatedPythonClient {
    child: ChildProcess,
    snapshots: SharedSnapshotStore,
    reader: tokio::process::ChildStdout,
    writer: tokio::process::ChildStdin,
    limits: FrameLimits,
    next_request_id: u64,
    next_callback_id: u64,
    local_to_remote: BTreeMap<u64, u64>,
    remote_to_local: BTreeMap<u64, (u64, WeakForeignRef)>,
    callbacks: BTreeMap<u64, Value>,
}

impl IsolatedPythonClient {
    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub async fn spawn(
        configuration: &PythonRuntimeConfiguration,
    ) -> Result<Self, PythonWireError> {
        let mut host_configuration = configuration.clone();
        if host_configuration.executable.is_none() {
            let installation = discover_python(&PythonDiscoveryRequest {
                executable: None,
                version: host_configuration.version,
                minimum_version: host_configuration.minimum_version,
                maximum_version: host_configuration.maximum_version,
                exact_version: host_configuration.exact_version,
                required_abi_tag: host_configuration.required_abi_tag.clone(),
                required_platform_tag: host_configuration.required_platform_tag.clone(),
            })
            .map_err(|error| wire_error("RunMat:Python:PythonDiscoveryError", error.to_string()))?;
            host_configuration.executable = Some(installation.executable);
            host_configuration.version = None;
        }
        let executable = std::env::current_exe()
            .map_err(|error| wire_error("RunMat:Python:HostExecutable", error.to_string()))?;
        let secret = SessionSecret::generate();
        let snapshots = SharedSnapshotStore::create()
            .map_err(|error| wire_error("RunMat:Python:ValueSnapshot", error.to_string()))?;
        let mut command = HostCommand::new(executable);
        command.arguments = vec![HiddenMode::ExtensionHost.marker().into()];
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        command
            .environment
            .insert(PYTHON_HOST_KIND_ENV.into(), PYTHON_HOST_KIND.into());
        command.environment.insert(
            PYTHON_HOST_CONFIG_ENV.into(),
            serde_json::to_string(&host_configuration)
                .map_err(|error| wire_error("RunMat:Python:HostConfig", error.to_string()))?,
        );
        command
            .environment
            .insert(PYTHON_HOST_SECRET_ENV.into(), secret.expose_hex());
        let snapshot_root = snapshots.root_path().to_str().ok_or_else(|| {
            wire_error(
                "RunMat:Python:ValueSnapshot",
                "shared snapshot root is not valid Unicode",
            )
        })?;
        command
            .environment
            .insert(PYTHON_HOST_SNAPSHOT_ROOT_ENV.into(), snapshot_root.into());
        let mut child = command
            .spawn()
            .await
            .map_err(|error| wire_error("RunMat:Python:HostSpawn", error.to_string()))?;
        let stdio = child
            .take_stdio()
            .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))?;
        let mut reader = stdio.stdout;
        let mut writer = stdio.stdin;
        let session = authenticate_driver(
            &mut reader,
            &mut writer,
            HostHandshake::new(
                PYTHON_HOST_PROTOCOL,
                PYTHON_HOST_SCHEMA_VERSION,
                PYTHON_HOST_MAX_MESSAGE_BYTES,
            ),
            &secret,
        )
        .await
        .map_err(|error| wire_error("RunMat:Python:HostAuthentication", error.to_string()))?;
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
            callbacks: BTreeMap::new(),
        })
    }

    pub async fn invoke(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        released_local_handles: Vec<u64>,
        nested: &NestedPythonQueue,
    ) -> Result<Vec<Value>, RuntimeError> {
        let request = self
            .request(call, released_local_handles)
            .await
            .map_err(runtime_error)?;
        let request_id = request.request_id;
        self.write(&PythonDriverMessage::Invoke(request))
            .await
            .map_err(runtime_error)?;
        self.wait_for_invocation(request_id, runtime, handles, host_identity, nested)
            .await
    }

    async fn request(
        &mut self,
        call: ForeignCall,
        released_local_handles: Vec<u64>,
    ) -> Result<PythonInvocationRequest, PythonWireError> {
        let request_id = self.take_request_id()?;
        let releases = self.translate_releases(released_local_handles);
        let mut arguments = Vec::with_capacity(call.arguments.len());
        for value in &call.arguments {
            arguments.push(self.encode_driver_value(value).await?);
        }
        Ok(PythonInvocationRequest {
            request_id,
            operation: call.symbol,
            arguments,
            requested_outputs: u32::try_from(call.requested_outputs).map_err(|_| {
                wire_error(
                    "RunMat:Python:InvalidCall",
                    "Python output count exceeds u32",
                )
            })?,
            releases,
        })
    }

    async fn wait_for_invocation(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> Result<Vec<Value>, RuntimeError> {
        loop {
            if runtime.cancellation().load(Ordering::Relaxed) {
                let _ = self.child.terminate_tree().await;
                return Err(runtime_error(wire_error(
                    "RunMat:Python:Cancelled",
                    "isolated Python invocation was cancelled",
                )));
            }
            let message = tokio::select! {
                message = read_message(&mut self.reader, self.limits) => message,
                _ = tokio::time::sleep(Duration::from_millis(10)) => continue,
            }
            .map_err(runtime_error)?;
            match message {
                PythonHostMessage::Invocation(result) if result.request_id == request_id => {
                    let values = result.outcome.map_err(runtime_error)?;
                    return self
                        .decode_driver_outputs(values, handles, host_identity)
                        .map_err(runtime_error);
                }
                PythonHostMessage::Callback(callback) if callback.request_id == request_id => {
                    let response = self
                        .handle_callback(callback, runtime.clone(), handles, host_identity, nested)
                        .await;
                    self.write(&PythonDriverMessage::CallbackResult(response))
                        .await
                        .map_err(runtime_error)?;
                }
                _ => {
                    return Err(runtime_error(wire_error(
                        "RunMat:Python:HostProtocol",
                        "host record does not match the active Python invocation",
                    )));
                }
            }
        }
    }

    async fn handle_callback(
        &mut self,
        callback: PythonCallbackRequest,
        runtime: RuntimeContext,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> PythonCallbackResult {
        let request_id = callback.request_id;
        let callback_id = callback.callback_id;
        let outcome = self
            .drive_callback(callback, runtime, handles, host_identity, nested)
            .await;
        PythonCallbackResult {
            request_id,
            callback_id,
            outcome: outcome.map_err(runtime_wire_error),
        }
    }

    async fn drive_callback(
        &mut self,
        callback: PythonCallbackRequest,
        runtime: RuntimeContext,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> Result<Vec<PythonWireValue>, RuntimeError> {
        let callable = self
            .callbacks
            .get(&callback.callback_id)
            .cloned()
            .ok_or_else(|| {
                runtime_error(wire_error(
                    "RunMat:Python:CallbackProtocol",
                    "host requested an unknown Python callback",
                ))
            })?;
        let mut arguments = Vec::with_capacity(callback.arguments.len());
        for value in callback.arguments {
            arguments.push(
                self.decode_driver_value(value, handles, host_identity)
                    .map_err(runtime_error)?,
            );
        }
        let requested_outputs = usize::try_from(callback.requested_outputs).map_err(|_| {
            runtime_error(wire_error(
                "RunMat:Python:CallbackProtocol",
                "Python callback output count is not representable",
            ))
        })?;
        let request = foreign_callback_request(&callable, arguments, requested_outputs)?;
        let callback_future = invoke_foreign_callback(runtime.clone(), request);
        tokio::pin!(callback_future);
        let sequence = loop {
            tokio::select! {
                result = &mut callback_future => break result?,
                _ = tokio::time::sleep(Duration::from_millis(1)) => {
                    let queued = nested.borrow_mut().pop_front();
                    if let Some(queued) = queued {
                        let result = self.invoke_nested(
                            runtime.clone(), queued.call, handles, host_identity, nested,
                        ).await;
                        let _ = queued.reply.send(result);
                    }
                }
            }
        };
        let values = crate::sequence::ResolveValueSequence::resolve(
            sequence,
            runmat_types::SequenceUse::SelectPrefix {
                count: requested_outputs,
            },
            crate::sequence::SequenceResolutionContext::default(),
        )?;
        let mut encoded = Vec::with_capacity(values.len());
        for value in values {
            let gathered = crate::gather_if_needed_async(&value).await?;
            encoded.push(
                self.encode_driver_value(&gathered)
                    .await
                    .map_err(runtime_error)?,
            );
        }
        Ok(encoded)
    }

    async fn invoke_nested(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> Result<Vec<Value>, RuntimeError> {
        let request = self
            .request(call, Vec::new())
            .await
            .map_err(runtime_error)?;
        let request_id = request.request_id;
        self.write(&PythonDriverMessage::Invoke(request))
            .await
            .map_err(runtime_error)?;
        Box::pin(self.wait_for_invocation(request_id, runtime, handles, host_identity, nested))
            .await
    }

    async fn encode_driver_value(
        &mut self,
        value: &Value,
    ) -> Result<PythonWireValue, PythonWireError> {
        if super::super::super::is_callable(value) {
            let id = self.next_callback_id;
            self.next_callback_id = self.next_callback_id.checked_add(1).ok_or_else(|| {
                wire_error("RunMat:Python:HostProtocol", "callback identity exhausted")
            })?;
            self.callbacks.insert(id, value.clone());
            return Ok(PythonWireValue::Callback { id });
        }
        match value {
            Value::Foreign(reference) => {
                let id = self
                    .local_to_remote
                    .get(&reference.handle)
                    .copied()
                    .ok_or_else(|| {
                        wire_error(
                            "RunMat:Python:StaleHandle",
                            "Python object does not belong to the active isolated host",
                        )
                    })?;
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
            _ => {
                let host = crate::gather_if_needed_async(value)
                    .await
                    .map_err(|error| wire_error("RunMat:Python:ValueGather", error.to_string()))?;
                encode_portable(&host, &self.snapshots).map(PythonWireValue::Portable)
            }
        }
    }

    fn decode_driver_value(
        &mut self,
        value: PythonWireValue,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<Value, PythonWireError> {
        match value {
            PythonWireValue::Portable(transfer) => decode_portable(&transfer, &self.snapshots),
            PythonWireValue::Foreign(remote) => {
                let reference = if let Some((_, reference)) = self.remote_to_local.get(&remote.id) {
                    reference.upgrade().ok_or_else(|| {
                        wire_error("RunMat:Python:StaleHandle", "Python object was released")
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
                            wire_error("RunMat:Python:HostUnavailable", error.to_string())
                        })?;
                    self.local_to_remote.insert(reference.handle, remote.id);
                    let weak = reference.downgrade().ok_or_else(|| {
                        wire_error(
                            "RunMat:Python:HostProtocol",
                            "Python object lease is missing",
                        )
                    })?;
                    self.remote_to_local
                        .insert(remote.id, (reference.handle, weak));
                    reference
                };
                Ok(Value::Foreign(reference))
            }
            PythonWireValue::Callback { .. } => Err(wire_error(
                "RunMat:Python:HostProtocol",
                "host returned a callback token as a value",
            )),
        }
    }

    fn decode_driver_outputs(
        &mut self,
        values: Vec<PythonWireValue>,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<Vec<Value>, PythonWireError> {
        values
            .into_iter()
            .map(|value| self.decode_driver_value(value, handles, host_identity))
            .collect()
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

    pub async fn shutdown(&mut self, released: Vec<u64>) -> Result<(), PythonWireError> {
        let request_id = self.take_request_id()?;
        let releases = self.translate_releases(released);
        self.write(&PythonDriverMessage::Shutdown {
            request_id,
            releases,
        })
        .await?;
        let message = tokio::time::timeout(
            DEFAULT_SHUTDOWN_TIMEOUT,
            read_message(&mut self.reader, self.limits),
        )
        .await
        .map_err(|_| {
            wire_error(
                "RunMat:Python:Timeout",
                "isolated Python shutdown timed out",
            )
        })??;
        let PythonHostMessage::Shutdown(PythonShutdownResult {
            request_id: response,
            outcome,
        }) = message
        else {
            return Err(wire_error(
                "RunMat:Python:HostProtocol",
                "host returned the wrong shutdown record",
            ));
        };
        if response != request_id {
            return Err(wire_error(
                "RunMat:Python:HostProtocol",
                "shutdown response identity does not match",
            ));
        }
        outcome?;
        self.callbacks.clear();
        let exit = self
            .child
            .wait()
            .await
            .map_err(|error| wire_error("RunMat:Python:HostExit", error.to_string()))?;
        if !exit.success {
            return Err(wire_error(
                "RunMat:Python:HostCrashed",
                format!("isolated Python host exited with code {:?}", exit.code),
            ));
        }
        Ok(())
    }

    async fn write(&mut self, message: &PythonDriverMessage) -> Result<(), PythonWireError> {
        write_message(&mut self.writer, message, self.limits).await
    }

    fn take_request_id(&mut self) -> Result<u64, PythonWireError> {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.checked_add(1).ok_or_else(|| {
            wire_error("RunMat:Python:HostProtocol", "request identity exhausted")
        })?;
        Ok(id)
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
            .unwrap_or("RunMat:Python:CallbackFailed")
            .into(),
        message: error.to_string(),
        python,
    }
}

pub(crate) fn runtime_error(error: PythonWireError) -> RuntimeError {
    let mut builder = build_runtime_error(error.message)
        .with_builtin("python")
        .with_identifier(error.identifier);
    if let Some(python) = error.python {
        builder = builder.with_source(python);
    }
    builder.build()
}

async fn read_message(
    reader: &mut (impl AsyncRead + Unpin),
    limits: FrameLimits,
) -> Result<PythonHostMessage, PythonWireError> {
    let payload = read_payload(reader, limits)
        .await
        .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))?;
    let message: PythonHostMessage = serde_json::from_slice(&payload)
        .map_err(|error| wire_error("RunMat:Python:HostProtocol", error.to_string()))?;
    message
        .validate()
        .map_err(|error| wire_error("RunMat:Python:HostProtocol", error))?;
    Ok(message)
}

async fn write_message(
    writer: &mut (impl AsyncWrite + Unpin),
    message: &PythonDriverMessage,
    limits: FrameLimits,
) -> Result<(), PythonWireError> {
    message
        .validate()
        .map_err(|error| wire_error("RunMat:Python:HostProtocol", error))?;
    let payload = serde_json::to_vec(message)
        .map_err(|error| wire_error("RunMat:Python:HostProtocol", error.to_string()))?;
    write_payload(writer, &payload, limits)
        .await
        .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))
}
