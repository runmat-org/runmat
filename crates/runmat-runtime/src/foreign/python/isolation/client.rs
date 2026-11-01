use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;
use std::time::Duration;

use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};
use runmat_process_host::ipc::private_control::PrivateControlListener;
use runmat_process_host::ipc::{
    authenticate_driver, read_payload, write_payload, FrameLimits, HostHandshake, SessionSecret,
};
use runmat_process_host::private_bytes::PrivateByteLease;
use runmat_process_host::shared_memory::{SharedSnapshotParent, SharedSnapshotStore};
use runmat_process_host::{
    BootstrapBoundChild, ContainedChildRetirement, HiddenMode, HostCommand, ProcessGroupPlacement,
    ProcessRetirementLease, ProcessSpawnEvidence,
};
use runmat_python::{discover_python, PythonDiscoveryRequest};
use runmat_value::{Value, WeakForeignRef};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::oneshot;

use super::{
    decode_portable, encode_portable, root_plain_value, wire_error, PythonCallbackRequest,
    PythonCallbackResult, PythonDriverMessage, PythonHostMessage, PythonInvocationRequest,
    PythonRemoteReference, PythonShutdownResult, PythonWireError, PythonWireValue,
    PYTHON_HOST_CONFIG_ENV, PYTHON_HOST_KIND, PYTHON_HOST_KIND_ENV, PYTHON_HOST_MAX_MESSAGE_BYTES,
    PYTHON_HOST_PROTOCOL, PYTHON_HOST_SCHEMA_VERSION, PYTHON_HOST_SECRET_ENV,
    PYTHON_HOST_SNAPSHOT_ROOT_ENV,
};
use crate::context::{ForeignCall, RuntimeContext};
use crate::execution::{RootedValueLease, RootedValueMember, RootedValues, RootedValuesBuilder};
use crate::foreign::isolation_io::{FramedReader, FramedWriter};
use crate::foreign::isolation_output::{self, Completion, HostOutput};
use crate::foreign::{
    invoke_foreign_callback, ForeignHandleRegistry, ForeignResourceMetadata,
    PythonRuntimeConfiguration,
};
use crate::{build_runtime_error, RuntimeError};

const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[path = "client/invocation_read.rs"]
mod invocation_read;
mod application_output;
use invocation_read::{read_invocation_message, InvocationRead};

pub struct NestedPythonCall {
    pub call: ForeignCall,
    pub reply: oneshot::Sender<Result<RootedValues, RuntimeError>>,
    pub(crate) ownership: Rc<()>,
}

pub type NestedPythonQueue = Rc<RefCell<VecDeque<NestedPythonCall>>>;

struct PythonChildResources {
    snapshots: SharedSnapshotStore,
    byte_leases: Vec<PrivateByteLease>,
}

impl ProcessRetirementLease for PythonChildResources {
    fn persist_unverified(&mut self, process_id: Option<u32>, reason: &str) -> Result<(), String> {
        let mut errors = Vec::new();
        if let Err(error) = self.snapshots.persist_unverified(process_id, reason) {
            errors.push(error);
        }
        for lease in &self.byte_leases {
            if let Err(error) = lease.persist_unverified(process_id, reason) {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

pub struct IsolatedPythonClient {
    child: BootstrapBoundChild<PythonChildResources>,
    reader: FramedReader,
    writer: FramedWriter,
    output: HostOutput,
    limits: FrameLimits,
    next_request_id: u64,
    next_callback_id: u64,
    local_to_remote: BTreeMap<u64, u64>,
    remote_to_local: BTreeMap<u64, (u64, WeakForeignRef)>,
    callbacks: BTreeMap<u64, crate::call::RetainedCallableTarget>,
    nested_aborted: bool,
    shutdown_progress: ShutdownProgress,
}

enum ShutdownProgress {
    Open,
    Attempted,
    Protocol(Result<(), PythonWireError>),
}

pub enum PythonClientRetirement {
    Retained {
        retirement: PythonWireError,
        prior_protocol: Option<PythonWireError>,
    },
    Retired {
        scope: ContainedChildRetirement,
        result: Result<(), PythonWireError>,
    },
}

impl IsolatedPythonClient {
    #[cfg(all(test, unix))]
    pub(crate) async fn pending_test_client() -> Self {
        Self::pending_test_client_with_evidence(ProcessSpawnEvidence::default()).await
    }

    #[cfg(all(test, unix))]
    pub(crate) async fn pending_test_client_with_evidence(
        unverified: ProcessSpawnEvidence,
    ) -> Self {
        let mut command = HostCommand::new("/bin/sh");
        command.arguments = vec!["-c".into(), "exec /bin/sleep 30".into()];
        let child = command
            .spawn_observed(&unverified)
            .await
            .expect("spawn owned pending host");
        let snapshots = SharedSnapshotStore::create().expect("test snapshots");
        let resources = PythonChildResources {
            snapshots,
            byte_leases: Vec::new(),
        };
        let mut child = BootstrapBoundChild::new(child, resources, unverified);
        let stdio = child.take_stdio().expect("take pending host stdio");
        Self {
            child,
            reader: FramedReader::Fixture(stdio.stdout),
            writer: FramedWriter::Fixture(stdio.stdin),
            output: HostOutput::default(),
            limits: FrameLimits {
                max_message_bytes: PYTHON_HOST_MAX_MESSAGE_BYTES,
            },
            next_request_id: 1,
            next_callback_id: 1,
            local_to_remote: BTreeMap::new(),
            remote_to_local: BTreeMap::new(),
            callbacks: BTreeMap::new(),
            nested_aborted: false,
            shutdown_progress: ShutdownProgress::Open,
        }
    }

    pub fn process_id(&self) -> Option<u32> {
        self.child.id()
    }

    pub async fn spawn(
        runtime: Option<&RuntimeContext>,
        configuration: &PythonRuntimeConfiguration,
        process_group_placement: ProcessGroupPlacement,
        snapshot_parent: Option<SharedSnapshotParent>,
        unverified: ProcessSpawnEvidence,
        byte_leases: Vec<PrivateByteLease>,
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
        let snapshots = match snapshot_parent.as_ref() {
            Some(parent) => SharedSnapshotStore::create_in_parent(parent),
            None => SharedSnapshotStore::create(),
        }
        .map_err(|error| wire_error("RunMat:Python:ValueSnapshot", error.to_string()))?;
        if (process_group_placement == ProcessGroupPlacement::JoinAncestorGroup)
            != snapshot_parent.is_some()
        {
            return Err(wire_error(
                "RunMat:Python:ValueSnapshot",
                "joined Python host requires parent-owned snapshot scratch",
            ));
        }
        let mut command = HostCommand::new(executable);
        command.process_group_placement = process_group_placement;
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
        let control = PrivateControlListener::bind()
            .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))?;
        command.stdio =
            runmat_process_host::StdioPolicy::PrivateControl(control.endpoint().clone());
        let child = match command.spawn_observed(&unverified).await {
            Ok(child) => child,
            Err(error) => {
                if unverified.is_unverified() {
                    let mut resources = PythonChildResources {
                        snapshots,
                        byte_leases,
                    };
                    if let Err(marker_error) = resources.persist_unverified(
                        None,
                        "Python child was created but containment could not be proved",
                    ) {
                        eprintln!("Python snapshot bytes retained without recovery marker: {marker_error}");
                    }
                }
                return Err(wire_error("RunMat:Python:HostSpawn", error.to_string()));
            }
        };
        let resources = PythonChildResources {
            snapshots,
            byte_leases,
        };
        let mut child = BootstrapBoundChild::new(child, resources, unverified);
        let output = child
            .take_application_output()
            .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))?;
        let control = tokio::time::timeout(DEFAULT_SHUTDOWN_TIMEOUT, control.accept())
            .await
            .map_err(|_| {
                wire_error(
                    "RunMat:Python:Timeout",
                    "Python host control bootstrap timed out",
                )
            })?
            .map_err(|error| wire_error("RunMat:Python:HostTransport", error.to_string()))?;
        let mut reader = FramedReader::Private(control.reader);
        let mut writer = FramedWriter::Private(control.writer);
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
            reader,
            writer,
            output: HostOutput::attached(output, runtime).map_err(runtime_wire_error)?,
            limits: session.limits,
            next_request_id: 1,
            next_callback_id: 1,
            local_to_remote: BTreeMap::new(),
            remote_to_local: BTreeMap::new(),
            callbacks: BTreeMap::new(),
            nested_aborted: false,
            shutdown_progress: ShutdownProgress::Open,
        })
    }

    async fn invoke_protocol(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        released_local_handles: Vec<u64>,
        nested: &NestedPythonQueue,
    ) -> Result<RootedValues, RuntimeError> {
        if !matches!(self.shutdown_progress, ShutdownProgress::Open) {
            return Err(runtime_error(wire_error(
                "RunMat:Python:RetirementUnverified",
                "isolated Python host has an unverified shutdown operation",
            )));
        }
        let _input_roots = RootedValueLease::for_values(call.arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
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
        let values = call
            .arguments
            .try_into_values()
            .map_err(|error| wire_error("RunMat:Python:ValueEncode", error.to_string()))?;
        for value in values {
            arguments.push(self.encode_driver_argument(value).await?);
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
    ) -> Result<RootedValues, RuntimeError> {
        loop {
            let message = match read_invocation_message(&mut self.reader, self.limits, &runtime)
                .await
                .map_err(runtime_error)?
            {
                InvocationRead::Message(message) => message,
                InvocationRead::Cancelled => {
                    // The unfinished read is gone before child retirement.
                    // This transport is never reused after cancellation.
                    let _ = self.child.terminate_contained_child().await;
                    return Err(runtime_error(wire_error(
                        "RunMat:Python:Cancelled",
                        "isolated Python invocation was cancelled",
                    )));
                }
            };
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
                    if self.nested_aborted {
                        return Err(runtime_error(wire_error(
                            "RunMat:Python:Cancelled",
                            "nested isolated Python requester was abandoned",
                        )));
                    }
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
        let mut arguments = RootedValuesBuilder::with_capacity(callback.arguments.len());
        for value in callback.arguments {
            arguments.push_rooted(
                self.decode_driver_value(value, handles, host_identity)
                    .map_err(runtime_error)?,
            );
        }
        let arguments = arguments
            .finish()
            .map_err(|error| error.into_runtime_error())?;
        let requested_outputs = usize::try_from(callback.requested_outputs).map_err(|_| {
            runtime_error(wire_error(
                "RunMat:Python:CallbackProtocol",
                "Python callback output count is not representable",
            ))
        })?;
        let sequence = async {
            let request = crate::foreign::foreign_callback_request_rooted(
                &runtime,
                &callable,
                arguments,
                requested_outputs,
            )?;
            let callback_future = invoke_foreign_callback(runtime.clone(), request);
            tokio::pin!(callback_future);
            loop {
                tokio::select! {
                    result = &mut callback_future => break result,
                    _ = tokio::time::sleep(Duration::from_millis(1)) => {
                        let queued = nested.borrow_mut().pop_front();
                        if let Some(queued) = queued {
                            self.dispatch_nested_call(
                                runtime.clone(), queued, handles, host_identity, nested,
                            ).await?;
                        }
                    }
                }
            }
        }
        .await?;
        let values = sequence.with_owned_sequence(|sequence| {
            let values = crate::sequence::ResolveValueSequence::resolve(
                sequence,
                runmat_types::SequenceUse::SelectPrefix {
                    count: requested_outputs,
                },
                crate::sequence::SequenceResolutionContext::default(),
            )?;
            crate::execution::RootedValues::from_output_storage_in_context(values, &runtime)
        })?;
        values
            .with_owned_output_storage_async(|values| async {
                let mut encoded = Vec::with_capacity(values.len());
                let values = values.try_into_owned().map_err(|_| {
                    crate::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?;
                for value in values {
                    crate::foreign::reject_scoped_callback(&value)?;
                    if super::super::super::is_callable(&value) {
                        encoded.push(
                            self.encode_driver_value(value)
                                .await
                                .map_err(runtime_error)?,
                        );
                    } else {
                        let gathered = crate::gather_if_needed_async(&value).await?;
                        let gathered = crate::execution::RootedValue::new(gathered)
                            .map_err(|error| error.into_runtime_error())?;
                        encoded.push(
                            gathered
                                .with_owned_async(|value| self.encode_driver_value(value))
                                .await
                                .map_err(runtime_error)?,
                        );
                    }
                }
                Ok(encoded)
            })
            .await
    }

    async fn invoke_nested(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> Result<RootedValues, RuntimeError> {
        let _input_roots = RootedValueLease::for_values(call.arguments.iter())
            .map_err(|error| error.into_runtime_error())?;
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

    async fn dispatch_nested_call(
        &mut self,
        runtime: RuntimeContext,
        mut queued: NestedPythonCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        nested: &NestedPythonQueue,
    ) -> Result<(), RuntimeError> {
        if queued.reply.is_closed() {
            return Ok(());
        }
        let result = tokio::select! {
            result = self.invoke_nested(
                runtime, queued.call, handles, host_identity, nested,
            ) => Some(result),
            _ = queued.reply.closed() => None,
        };
        let Some(result) = result else {
            self.nested_aborted = true;
            return Err(runtime_error(wire_error(
                "RunMat:Python:Cancelled",
                "nested isolated Python requester was abandoned",
            )));
        };
        if queued.reply.is_closed() {
            self.nested_aborted = true;
            return Err(runtime_error(wire_error(
                "RunMat:Python:Cancelled",
                "nested isolated Python requester was abandoned",
            )));
        }
        let _ = queued.reply.send(result);
        Ok(())
    }

    async fn encode_driver_value(
        &mut self,
        value: Value,
    ) -> Result<PythonWireValue, PythonWireError> {
        if super::super::super::is_callable(&value) {
            let id = self.next_callback_id;
            self.next_callback_id = self.next_callback_id.checked_add(1).ok_or_else(|| {
                wire_error("RunMat:Python:HostProtocol", "callback identity exhausted")
            })?;
            let generation = runmat_value::SharedValueGeneration::unbounded(value);
            let callable = crate::execution::RetainedValueReadLease::retain_shared_generation(
                generation, None,
            )
            .map_err(|error| wire_error("RunMat:Python:ValueEncode", error.to_string()))?;
            self.callbacks
                .insert(id, crate::call::RetainedCallableTarget::Direct(callable));
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
            other => {
                let host = crate::gather_if_needed_async(&other)
                    .await
                    .map_err(|error| wire_error("RunMat:Python:ValueGather", error.to_string()))?;
                encode_portable(&host, &self.child.lease().snapshots).map(PythonWireValue::Portable)
            }
        }
    }

    async fn encode_driver_argument(
        &mut self,
        value: crate::call::RootedArgumentValue,
    ) -> Result<PythonWireValue, PythonWireError> {
        crate::foreign::reject_scoped_callback(value.value())
            .map_err(|error| wire_error("RunMat:Python:ValueEncode", error.to_string()))?;
        if super::super::super::is_callable(value.value()) {
            let id = self.next_callback_id;
            self.next_callback_id = self.next_callback_id.checked_add(1).ok_or_else(|| {
                wire_error("RunMat:Python:HostProtocol", "callback identity exhausted")
            })?;
            let owner = match value {
                crate::call::RootedArgumentValue::Owned(value) => {
                    let generation = runmat_value::SharedValueGeneration::unbounded(value);
                    crate::call::RetainedCallableTarget::Direct(
                        crate::execution::RetainedValueReadLease::retain_shared_generation(
                            generation, None,
                        )
                        .map_err(|error| {
                            wire_error("RunMat:Python:ValueEncode", error.to_string())
                        })?,
                    )
                }
                crate::call::RootedArgumentValue::Captured(value) => {
                    crate::call::RetainedCallableTarget::Captured(value)
                }
            };
            self.callbacks.insert(id, owner);
            return Ok(PythonWireValue::Callback { id });
        }
        match value.into_owned() {
            Ok(value) => self.encode_driver_value(value).await,
            Err(value) => {
                let value = crate::gather_if_needed_async(value.value())
                    .await
                    .map_err(|error| wire_error("RunMat:Python:ValueGather", error.to_string()))?;
                encode_portable(&value, &self.child.lease().snapshots)
                    .map(PythonWireValue::Portable)
            }
        }
    }

    fn decode_driver_value(
        &mut self,
        value: PythonWireValue,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
    ) -> Result<RootedValueMember, PythonWireError> {
        match value {
            PythonWireValue::Portable(transfer) => {
                decode_portable(&transfer, &self.child.lease().snapshots)
            }
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
                root_plain_value(Value::Foreign(reference))
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
    ) -> Result<RootedValues, PythonWireError> {
        let mut decoded = RootedValuesBuilder::with_capacity(values.len());
        for value in values {
            decoded.push_rooted(self.decode_driver_value(value, handles, host_identity)?);
        }
        decoded
            .finish()
            .map_err(|error| wire_error("RunMat:Python:ValueDecode", error.to_string()))
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

    async fn shutdown_protocol(&mut self, released: Vec<u64>) -> Result<(), PythonWireError> {
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
        let exit = tokio::time::timeout(DEFAULT_SHUTDOWN_TIMEOUT, self.child.wait())
            .await
            .map_err(|_| {
                wire_error(
                    "RunMat:Python:Timeout",
                    "isolated Python host exit timed out",
                )
            })?
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

fn console_transport_error(error: RuntimeError) -> RuntimeError {
    build_runtime_error("isolated Python application output could not be delivered")
        .with_identifier("RunMat:Python:HostTransport")
        .with_source(error)
        .build()
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

#[cfg(all(test, unix))]
mod tests;
