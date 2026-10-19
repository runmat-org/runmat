use std::collections::BTreeMap;
use std::ops::Deref;
use std::time::Duration;

use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::{SharedSnapshotParent, SharedSnapshotStore};
use runmat_process_host::{
    BootstrapBoundChild, ContainedChildRetirement, HiddenMode, HostCommand, ProcessGroupPlacement,
    ProcessRetirementLease, ProcessSpawnEvidence,
};
use runmat_value::{Value, WeakForeignRef};
use tokio::io::{AsyncRead, AsyncWrite};

use super::{
    decode_portable, encode_portable, root_plain_value, wire_error, NativeCallbackRequest,
    NativeCallbackResult, NativeDriverMessage, NativeHostMessage, NativeInvocationRequest,
    NativeProjectLoadRequest, NativeRemoteReference, NativeShutdownResult, NativeWireError,
    NativeWireValue, NATIVE_FFI_HOST_FRONTEND_ENV, NATIVE_FFI_HOST_KIND, NATIVE_FFI_HOST_KIND_ENV,
    NATIVE_FFI_HOST_MAX_MESSAGE_BYTES, NATIVE_FFI_HOST_PROTOCOL, NATIVE_FFI_HOST_SCHEMA_VERSION,
    NATIVE_FFI_HOST_SECRET_ENV, NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{ForeignCall, RuntimeContext};
use crate::execution::{RootedValueLease, RootedValueMember, RootedValues, RootedValuesBuilder};
use crate::foreign::{ForeignErrorKind, ForeignHandleRegistry, ForeignResourceMetadata};
use crate::{build_runtime_error, foreign::foreign_error, RuntimeError};

const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

struct NativeChildResources {
    snapshots: SharedSnapshotStore,
    project_bytes: Option<runmat_process_host::private_bytes::PrivateByteLease>,
}

impl Deref for NativeChildResources {
    type Target = SharedSnapshotStore;

    fn deref(&self) -> &Self::Target {
        &self.snapshots
    }
}

impl ProcessRetirementLease for NativeChildResources {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        let snapshot_result = self.snapshots.persist_unverified(process_id, reason);
        let project_result = self
            .project_bytes
            .as_ref()
            .map(|lease| lease.persist_unverified(process_id, reason))
            .unwrap_or(Ok(()));
        snapshot_result.and(project_result)
    }
}

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
    child: BootstrapBoundChild<NativeChildResources>,
    reader: tokio::process::ChildStdout,
    writer: tokio::process::ChildStdin,
    limits: FrameLimits,
    next_request_id: u64,
    next_callback_id: u64,
    local_to_remote: BTreeMap<u64, u64>,
    remote_to_local: BTreeMap<u64, (u64, WeakForeignRef)>,
    shutdown_progress: ShutdownProgress,
    user_state: NativeUserState,
}

#[derive(Clone, Copy)]
enum NativeUserState {
    Known(bool),
    ProjectLoadUnverifiedWithoutUser,
    UserInvocationUnverified,
}

enum ShutdownProgress {
    Open,
    Attempted,
    Protocol(Result<(), NativeWireError>),
}

pub enum NativeClientRetirement {
    Retained {
        retirement: NativeWireError,
        prior_protocol: Option<NativeWireError>,
    },
    Retired {
        scope: ContainedChildRetirement,
        result: Result<(), NativeWireError>,
    },
}

impl IsolatedNativeFfiClient {
    pub async fn load_project(
        &mut self,
        runtime: RuntimeContext,
        mut request: NativeProjectLoadRequest,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        timeout: Option<Duration>,
    ) -> Result<(), RuntimeError> {
        let request_id = self.take_request_id().map_err(runtime_error)?;
        request.request_id = request_id;
        request
            .validate()
            .map_err(|error| runtime_error(wire_error("RunMat:NativeFFI:ProjectLoad", error)))?;
        self.write(&NativeDriverMessage::ProjectLoad(request))
            .await
            .map_err(runtime_error)?;
        let wait =
            self.wait_for_invocation(request_id, runtime, BTreeMap::new(), handles, host_identity);
        let outcome = if let Some(timeout) = timeout {
            match tokio::time::timeout(timeout, wait).await {
                Ok(outcome) => outcome,
                Err(_) => {
                    let _ = self.child.terminate_contained_child().await;
                    return Err(runtime_error(wire_error(
                        "RunMat:NativeFFI:ProjectLoad",
                        "isolated native project load timed out",
                    )));
                }
            }
        } else {
            wait.await
        }?;
        if !outcome.values().is_empty() {
            return Err(runtime_error(wire_error(
                "RunMat:NativeFFI:ProjectLoad",
                "isolated native project load returned values",
            )));
        }
        Ok(())
    }

    pub fn begin_invocation(&mut self) -> Result<(), NativeWireError> {
        if !matches!(self.shutdown_progress, ShutdownProgress::Open) {
            return Err(wire_error(
                "RunMat:NativeFFI:RetirementUnverified",
                "isolated native-library host has an unverified prior operation",
            ));
        }
        // A cancelled invocation may have written to the child. Until its
        // result returns, shutdown must not send another ambiguous command.
        self.shutdown_progress = ShutdownProgress::Attempted;
        self.user_state = NativeUserState::UserInvocationUnverified;
        Ok(())
    }

    pub fn begin_project_load(&mut self) -> Result<(), NativeWireError> {
        if !matches!(self.shutdown_progress, ShutdownProgress::Open) {
            return Err(wire_error(
                "RunMat:NativeFFI:RetirementUnverified",
                "isolated native-library host has an unverified prior operation",
            ));
        }
        self.shutdown_progress = ShutdownProgress::Attempted;
        self.user_state = match self.user_state {
            NativeUserState::Known(false) => NativeUserState::ProjectLoadUnverifiedWithoutUser,
            _ => NativeUserState::UserInvocationUnverified,
        };
        Ok(())
    }

    pub fn permits_project_retirement(&self) -> bool {
        matches!(
            self.user_state,
            NativeUserState::Known(false) | NativeUserState::ProjectLoadUnverifiedWithoutUser
        )
    }

    pub async fn spawn(
        process_group_placement: ProcessGroupPlacement,
        unverified: ProcessSpawnEvidence,
        snapshot_parent: Option<SharedSnapshotParent>,
        project_bytes: Option<runmat_process_host::private_bytes::PrivateByteLease>,
    ) -> Result<Self, NativeWireError> {
        let executable = std::env::current_exe()
            .map_err(|error| wire_error("RunMat:NativeFFI:HostExecutable", error.to_string()))?;
        let secret = SessionSecret::generate();
        let snapshots = match snapshot_parent.as_ref() {
            Some(parent) => SharedSnapshotStore::create_in_parent(parent),
            None => SharedSnapshotStore::create(),
        }
        .map_err(|error| wire_error("RunMat:NativeFFI:ValueSnapshot", error.to_string()))?;
        let mut resources = NativeChildResources {
            snapshots,
            project_bytes,
        };
        if (process_group_placement == ProcessGroupPlacement::JoinAncestorGroup)
            != snapshot_parent.is_some()
        {
            return Err(wire_error(
                "RunMat:NativeFFI:ValueSnapshot",
                "joined native host requires parent-owned snapshot scratch",
            ));
        }
        let mut command = HostCommand::new(executable);
        command.process_group_placement = process_group_placement;
        command.arguments = vec![HiddenMode::ExtensionHost.marker().into()];
        command.environment_policy =
            EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime());
        command
            .environment
            .insert(NATIVE_FFI_HOST_KIND_ENV.into(), NATIVE_FFI_HOST_KIND.into());
        let frontend = resolve_frontend();
        let frontend = frontend.to_str().ok_or_else(|| {
            wire_error(
                "RunMat:NativeFFI:HostFrontend",
                "native FFI compiler frontend path is not valid UTF-8",
            )
        })?;
        command
            .environment
            .insert(NATIVE_FFI_HOST_FRONTEND_ENV.into(), frontend.into());
        command
            .environment
            .insert(NATIVE_FFI_HOST_SECRET_ENV.into(), secret.expose_hex());
        let snapshot_root = resources.root_path().to_str().ok_or_else(|| {
            wire_error(
                "RunMat:NativeFFI:ValueSnapshot",
                "shared snapshot root is not valid Unicode",
            )
        })?;
        command.environment.insert(
            NATIVE_FFI_HOST_SNAPSHOT_ROOT_ENV.into(),
            snapshot_root.into(),
        );
        let child = match command.spawn_observed(&unverified).await {
            Ok(child) => child,
            Err(error) => {
                if unverified.is_unverified() {
                    if let Err(marker_error) = resources.persist_unverified(
                        None,
                        "native FFI child was created but containment could not be proved",
                    ) {
                        eprintln!("native FFI snapshot bytes retained without recovery marker: {marker_error}");
                    }
                }
                return Err(wire_error("RunMat:NativeFFI:HostSpawn", error.to_string()));
            }
        };
        let mut child = BootstrapBoundChild::new(child, resources, unverified);
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
            reader,
            writer,
            limits: session.limits,
            next_request_id: 1,
            next_callback_id: 1,
            local_to_remote: BTreeMap::new(),
            remote_to_local: BTreeMap::new(),
            shutdown_progress: ShutdownProgress::Open,
            user_state: NativeUserState::Known(false),
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
    ) -> Result<RootedValues, RuntimeError> {
        let _input_roots = RootedValueLease::for_values(call.arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
        let request_id = self.take_request_id().map_err(runtime_error)?;
        let releases = self.translate_releases(released_local_handles);
        let mut callbacks = BTreeMap::new();
        let mut arguments = Vec::with_capacity(call.arguments.len());
        for value in call
            .arguments
            .try_into_values()
            .map_err(|error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string()))?
        {
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
                    let _ = self.child.terminate_contained_child().await;
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
            let _ = self.child.terminate_contained_child().await;
        }
        outcome
    }

    async fn wait_for_invocation(
        &mut self,
        request_id: u64,
        runtime: RuntimeContext,
        callbacks: BTreeMap<u64, crate::call::RetainedCallableTarget>,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<RootedValues, RuntimeError> {
        loop {
            if runtime.is_cancelled() {
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
                    // Only the exact terminal host record proves this request
                    // cannot be executing when another command is sent.
                    self.shutdown_progress = ShutdownProgress::Open;
                    self.user_state = NativeUserState::Known(result.user_state_present);
                    let values = result.outcome.map_err(runtime_error)?;
                    return self
                        .decode_driver_outputs(values, handles, host_identity)
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
        callbacks: &BTreeMap<u64, crate::call::RetainedCallableTarget>,
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
            let mut arguments = RootedValuesBuilder::with_capacity(callback.arguments.len());
            for value in callback.arguments {
                match value {
                    NativeWireValue::Portable(transfer) => {
                        arguments.push_rooted(decode_portable(&transfer, self.child.lease())?);
                    }
                    _ => {
                        return Err(wire_error(
                            "RunMat:NativeFFI:CallbackProtocol",
                            "native callback arguments must be portable values",
                        ))
                    }
                }
            }
            let arguments = arguments
                .finish()
                .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))?;
            let request =
                crate::foreign::foreign_callback_request_rooted(&runtime, &callable, arguments, 1)
                    .map_err(|error| {
                        wire_error("RunMat:NativeFFI:CallbackFailed", error.to_string())
                    })?;
            let sequence = crate::foreign::invoke_foreign_callback(runtime.clone(), request)
                .await
                .map_err(|error| {
                    wire_error("RunMat:NativeFFI:CallbackFailed", error.to_string())
                })?;
            let result = sequence
                .value()
                .as_slice()
                .first()
                .cloned()
                .ok_or_else(|| {
                    wire_error(
                        "RunMat:NativeFFI:CallbackFailed",
                        "callback returned no value",
                    )
                })?;
            let host_value = crate::gather_if_needed_async(&result)
                .await
                .map_err(|error| wire_error("RunMat:NativeFFI:ValueGather", error.to_string()))?;
            let encoded = encode_portable(&host_value, self.child.lease())
                .map(NativeWireValue::Portable)
                .map(|value| vec![value]);
            encoded
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
        value: crate::call::RootedArgumentValue,
        callbacks: &mut BTreeMap<u64, crate::call::RetainedCallableTarget>,
    ) -> Result<NativeWireValue, NativeWireError> {
        if super::super::is_callable(value.value()) {
            let id = self.next_callback_id;
            self.next_callback_id = self.next_callback_id.checked_add(1).ok_or_else(|| {
                wire_error(
                    "RunMat:NativeFFI:HostProtocol",
                    "callback identity exhausted",
                )
            })?;
            let owner = match value {
                crate::call::RootedArgumentValue::Owned(value) => {
                    let generation = runmat_value::SharedValueGeneration::unbounded(value);
                    crate::call::RetainedCallableTarget::Direct(
                        crate::execution::RetainedValueReadLease::retain_shared_generation(
                            generation, None,
                        )
                        .map_err(|error| {
                            wire_error("RunMat:NativeFFI:ValueEncode", error.to_string())
                        })?,
                    )
                }
                crate::call::RootedArgumentValue::Captured(value) => {
                    crate::call::RetainedCallableTarget::Captured(value)
                }
            };
            callbacks.insert(id, owner);
            return Ok(NativeWireValue::Callback { id });
        }
        match value.value() {
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
            Value::OutputList(_) => Err(wire_error(
                "RunMat:TransientSequenceNotPortable",
                "transient output sequences cannot be encoded as nested native FFI values",
            )),
            _ => {
                let host_value =
                    crate::gather_if_needed_async(value.value())
                        .await
                        .map_err(|error| {
                            wire_error("RunMat:NativeFFI:ValueGather", error.to_string())
                        })?;
                encode_portable(&host_value, self.child.lease()).map(NativeWireValue::Portable)
            }
        }
    }

    fn decode_driver_value(
        &mut self,
        value: NativeWireValue,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<RootedValueMember, NativeWireError> {
        match value {
            NativeWireValue::Portable(transfer) => decode_portable(&transfer, self.child.lease()),
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
                root_plain_value(Value::Foreign(reference))
            }
            NativeWireValue::Callback { .. } => Err(wire_error(
                "RunMat:NativeFFI:HostProtocol",
                "host returned a callback token as a value",
            )),
        }
    }

    fn decode_driver_outputs(
        &mut self,
        values: Vec<NativeWireValue>,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<RootedValues, NativeWireError> {
        let mut decoded = RootedValuesBuilder::with_capacity(values.len());
        for value in values {
            decoded.push_rooted(self.decode_driver_value(value, handles, host_identity)?);
        }
        decoded
            .finish()
            .map_err(|error| wire_error("RunMat:NativeFFI:ValueDecode", error.to_string()))
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

    pub async fn shutdown_retirement(
        &mut self,
        released_local_handles: Vec<u64>,
    ) -> NativeClientRetirement {
        if matches!(self.shutdown_progress, ShutdownProgress::Open) {
            // If this future is cancelled after a write, retry must not send
            // another ambiguous Shutdown record to the same child.
            self.shutdown_progress = ShutdownProgress::Attempted;
            let result = self.shutdown_protocol(released_local_handles).await;
            self.shutdown_progress = ShutdownProgress::Protocol(result);
        }
        match self.child.terminate_contained_child().await {
            Ok(scope) => {
                match std::mem::replace(&mut self.shutdown_progress, ShutdownProgress::Attempted) {
                    ShutdownProgress::Protocol(result) => {
                        NativeClientRetirement::Retired { scope, result }
                    }
                    ShutdownProgress::Attempted => NativeClientRetirement::Retired {
                        scope,
                        result: if matches!(
                            self.user_state,
                            NativeUserState::ProjectLoadUnverifiedWithoutUser
                        ) {
                            Ok(())
                        } else {
                            Err(wire_error(
                                "RunMat:NativeFFI:ShutdownInterrupted",
                                "isolated native-library operation was interrupted before retirement",
                            ))
                        },
                    },
                    ShutdownProgress::Open => unreachable!("shutdown was attempted before proof"),
                }
            }
            Err(error) => NativeClientRetirement::Retained {
                retirement: wire_error("RunMat:NativeFFI:RetirementUnverified", error.to_string()),
                // Keep the original phase for a same-child retry. The clone
                // supplies diagnostic ownership without resending Shutdown.
                prior_protocol: match &self.shutdown_progress {
                    ShutdownProgress::Protocol(Err(prior)) => Some(prior.clone()),
                    _ => None,
                },
            },
        }
    }

    async fn shutdown_protocol(
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
        let exit = tokio::time::timeout(DEFAULT_SHUTDOWN_TIMEOUT, self.child.wait())
            .await
            .map_err(|_| {
                wire_error(
                    "RunMat:NativeFFI:Timeout",
                    "isolated native-library host exit timed out",
                )
            })?
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

fn resolve_frontend() -> std::path::PathBuf {
    let executable = if cfg!(windows) { "clang.exe" } else { "clang" };
    std::env::var_os("PATH")
        .and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|directory| directory.join(executable))
                .find(|candidate| candidate.is_file())
        })
        .unwrap_or_else(|| executable.into())
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
