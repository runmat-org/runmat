use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{SinkExt as _, StreamExt as _};
use runmat_analysis_execution::StructuralStudyWorkloads;
use runmat_analysis_fea::execution::{
    AnalysisWorkKind, StructuralTransientBlockPlan, StructuralTransientWorkload,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::resource::ResourceInventory;
use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointIdentitySigner, EndpointRecipientKey, ExecutionTrustTier,
    WorkerTreeRetirementEvidenceV1,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::encryption::{
    encode_run_key_envelope, EncryptionPurpose, PortableExecutionEncryption, RunKeyMaterial,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, ExecutionBundleBuilder, ProgramExecutionDescriptor, ProgramExecutionInputs,
    ProgramExecutionObject, ProgramExecutionObjectClosure, ProgramExecutionObjectLocation,
    ProgramExecutionOutputs, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_execution_runner::WorkerSpec;
use runmat_execution_transport_native::control::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverAuthority,
    DriverBootstrap, DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition,
    DriverWorkerAllocation, DriverWorkerPool, ResourceRequest, StoreDriverArtifact,
    StoredDriverArtifact, WorkerRetirementLookup,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{MeshingRequest, MetricTensor3};

use super::checkpoint::RemoteDriverCheckpoint;
use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, project_revision_identity, seal_object};
use super::worker_attempt_journal::WorkerAttemptJournalOwner;
use super::worker_entry::RemoteWorkerRelayRequest;
use super::worker_env::session_id;
use crate::{
    import_exact_structural_study_result, native_meshing_kernel_dispatcher,
    prepare_exact_meshing_root, prepare_exact_structural_study_root, ExactMeshingRootPreparation,
    ExactStructuralStudyAnalysisResult, ExactStructuralStudyRootPreparation,
    NativeMeshingHostLimits,
};

const BOX: &[u8] = include_bytes!("../../../runmat-geometry/io/tests/fixtures/box.brep");
const RUN_ID: &str = "driver-replacement-exact-study";
const DRIVER_ALLOCATION: &str = "driver-allocation";
const ACTUAL_DRIVER_TEST_STACK_BYTES: usize = 16 * 1024 * 1024;

fn worker_journal(
    root: &Path,
    allocation: &DriverWorkerAllocation,
    fence: u64,
    clock_domain_nonce: [u8; 16],
) -> WorkerAttemptJournalOwner {
    let directory = root.join(format!("worker-{fence}-attempt-journal"));
    std::fs::create_dir(&directory).unwrap();
    WorkerAttemptJournalOwner {
        directory,
        run_id: RUN_ID.into(),
        allocation_lease_id: allocation.allocation_lease_id.clone(),
        allocation_fencing_token: allocation.fencing_token,
        driver_fencing_token: fence,
        process_launch_nonce: loop {
            let nonce: [u8; 16] = rand::random();
            if nonce.iter().any(|value| *value != 0) {
                break nonce;
            }
        },
        clock_domain_nonce,
    }
}

#[derive(Clone)]
struct StoredArtifact {
    kind: String,
    ciphertext: Vec<u8>,
}

struct DurableState {
    artifacts: BTreeMap<String, StoredArtifact>,
    initial_artifacts: Vec<DriverArtifactDownload>,
    latest_checkpoint: Option<String>,
    next_artifact: u64,
    driver_interruption_triggered: bool,
    second_driver_interruption_triggered: bool,
    interrupted_committed_task: Option<runmat_execution::TaskId>,
    cancellation_heartbeat_observed: bool,
    transitions: Vec<(u64, DriverRunTarget, Option<String>)>,
    worker_authorizations: Vec<(u64, String)>,
    bootstrap_checkpoints: Vec<(u64, Option<String>)>,
    retirements: BTreeMap<String, WorkerTreeRetirementEvidenceV1>,
}

struct DurableControl {
    project_revision: String,
    endpoint_fingerprint: String,
    run_key_envelope: Vec<u8>,
    run_key: RunKeyMaterial,
    resources: ResourceRequest,
    workers: BTreeMap<u64, DriverWorkerAllocation>,
    worker_identity_secrets: BTreeMap<u64, [u8; 32]>,
    interruption: ReplacementInterruption,
    interrupted_target: super::worker_execution::HeldAttemptTarget,
    state: Mutex<DurableState>,
    recovery_committed: tokio::sync::Notify,
    clock_control: Mutex<Option<Arc<observer::ClockControl>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplacementInterruption {
    RecoveryClosure,
    PoolTaskCommitted,
    ReplayableExecutionStart,
    AnalysisCommitted,
    AnalysisCheckpointAckLost,
    AnalysisCheckpointRejectedBeforeStore,
    WorkloadAttemptActive,
    WorkloadAttemptCancelledAfterRestore,
    WorkloadDeadlineExpiredAfterRestore,
    WorkloadRetryBudgetExhaustedAfterRestore,
}

impl DurableControl {
    fn record_retired_worker(
        &self,
        fence: u64,
        process_id: u32,
        journal: &WorkerAttemptJournalOwner,
        retired: runmat_process_host::clock::HostMonotonicReading,
    ) {
        let allocation = self
            .workers
            .get(&fence)
            .expect("original worker allocation");
        let endpoint = allocation
            .endpoint_identity
            .as_ref()
            .expect("signed worker endpoint");
        let signer =
            EndpointIdentitySigner::from_secret(self.worker_identity_secrets[&fence]).unwrap();
        assert_eq!(signer.fingerprint(), endpoint.identity_fingerprint);
        assert_eq!(journal.driver_fencing_token, fence);
        assert_eq!(journal.allocation_lease_id, allocation.allocation_lease_id);
        let mut encrypted_attempt_receipts = std::fs::read_dir(&journal.directory)
            .unwrap()
            .map(|file| {
                let file = file.unwrap();
                assert!(file.file_type().unwrap().is_file());
                let bytes = std::fs::read(file.path()).unwrap();
                assert!(
                    bytes.len()
                        <= runmat_execution::security::MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES
                );
                let object = runmat_execution_artifact::encryption::decode_encrypted_run_object(
                    &bytes,
                    runmat_execution::security::MAX_RETIREMENT_ENCRYPTED_RECEIPT_BYTES,
                )
                .unwrap();
                assert_eq!(object.context.run_identity, RUN_ID);
                assert_eq!(object.context.purpose, EncryptionPurpose::RetirementReceipt);
                assert!(object.context.task_identity.is_none());
                assert!(object.context.attempt_identity.is_none());
                (file.file_name(), URL_SAFE_NO_PAD.encode(bytes))
            })
            .collect::<Vec<_>>();
        encrypted_attempt_receipts.sort_by(|left, right| left.0.cmp(&right.0));
        assert!(
            encrypted_attempt_receipts.len() <= runmat_execution::security::MAX_RETIREMENT_ATTEMPTS
        );
        let mut evidence = WorkerTreeRetirementEvidenceV1 {
            schema_version: 1,
            org_id: endpoint.org_id.clone(),
            cluster_id: endpoint.cluster_id.clone(),
            node_id: endpoint.node_id.clone(),
            run_id: RUN_ID.into(),
            allocation_lease_id: allocation.allocation_lease_id.clone(),
            allocation_fencing_token: allocation.fencing_token,
            driver_fencing_token: fence,
            process_id,
            process_launch_nonce: journal.process_launch_nonce,
            retired_at_unix_millis: now_millis(),
            clock_domain_nonce: journal.clock_domain_nonce,
            retired_monotonic_ticks: retired.ticks,
            ticks_per_second: retired.ticks_per_second,
            encrypted_attempt_receipts: encrypted_attempt_receipts
                .into_iter()
                .map(|(_, bytes)| bytes)
                .collect(),
            identity_public_key: endpoint.identity_public_key.clone(),
            identity_fingerprint: endpoint.identity_fingerprint.clone(),
            signature: Vec::new(),
        };
        signer.sign_worker_tree_retirement(&mut evidence).unwrap();
        let mut state = self.state.lock().unwrap();
        assert!(state
            .retirements
            .insert(allocation.allocation_lease_id.clone(), evidence)
            .is_none());
    }
    fn interrupted_analysis_kind(&self) -> AnalysisWorkKind {
        match self.interrupted_target {
            super::worker_execution::HeldAttemptTarget::Analysis(kind) => kind,
            super::worker_execution::HeldAttemptTarget::ExactMeshingAdmission => {
                panic!("analysis interruption requires an analysis work kind")
            }
        }
    }

    fn interrupted_target(&self) -> super::worker_execution::HeldAttemptTarget {
        self.interrupted_target
    }

    fn active_interrupted_task(
        &self,
        state: &DurableState,
        checkpoint: &RemoteDriverCheckpoint,
    ) -> Option<runmat_execution::TaskId> {
        match self.interrupted_target() {
            super::worker_execution::HeldAttemptTarget::Analysis(kind) => {
                analysis_task(self, state, checkpoint, kind, true)
            }
            super::worker_execution::HeldAttemptTarget::ExactMeshingAdmission => {
                stored_pool_checkpoint(self, state, checkpoint).and_then(|(pool, _)| {
                    pool.test_snapshot().tasks.iter().find_map(|(id, task)| {
                        let program = pool.test_program(*id)?;
                        (task.committed.is_none()
                            && task.active_attempt.is_some()
                            && program.artifact.form
                                == runmat_execution_artifact::ExecutableForm::MeshingWorkload
                            && program.recipe.execution_mode
                                == crate::EXACT_MESHING_ROOT_EXECUTION_MODE)
                            .then_some(*id)
                    })
                })
            }
        }
    }
}

/// Inspect only authenticated, already-stored checkpoint objects. Never infer a
/// completed thermal phase from wall time or from an outer run status.
fn stored_pool_checkpoint(
    control: &DurableControl,
    state: &DurableState,
    checkpoint: &RemoteDriverCheckpoint,
) -> Option<(super::RemotePoolCheckpoint, ProgramExecutionObjectClosure)> {
    let catalog = checkpoint.pool_catalog.as_ref()?;
    let catalog_artifact = state.artifacts.get(&catalog.artifact_id)?;
    let catalog_bytes = open_object(
        &control.run_key,
        &catalog_artifact.ciphertext,
        RUN_ID,
        EncryptionPurpose::Result,
    )
    .ok()?;
    ProgramExecutionObject {
        reference: catalog.reference.clone(),
        bytes: catalog_bytes.clone(),
    }
    .validate()
    .ok()?;
    let closure = ProgramExecutionObjectClosure::from_canonical_bytes(&catalog_bytes).ok()?;
    let root = closure
        .objects
        .iter()
        .find(|location| location.reference == closure.root)?;
    let root_artifact = state.artifacts.get(&root.artifact_id)?;
    let root_bytes = open_object(
        &control.run_key,
        &root_artifact.ciphertext,
        RUN_ID,
        EncryptionPurpose::Result,
    )
    .ok()?;
    ProgramExecutionObject {
        reference: root.reference.clone(),
        bytes: root_bytes.clone(),
    }
    .validate()
    .ok()?;
    let pool = super::RemotePoolCheckpoint::decode(&root_bytes).ok()?;
    Some((pool, closure))
}

fn stored_driver_checkpoint(
    control: &DurableControl,
    state: &DurableState,
    artifact_id: &str,
) -> RemoteDriverCheckpoint {
    let artifact = state
        .artifacts
        .get(artifact_id)
        .expect("checkpoint artifact");
    let bytes = open_object(
        &control.run_key,
        &artifact.ciphertext,
        RUN_ID,
        EncryptionPurpose::Checkpoint,
    )
    .expect("authenticated checkpoint");
    RemoteDriverCheckpoint::from_canonical_bytes(&bytes).expect("valid checkpoint")
}

fn analysis_task(
    control: &DurableControl,
    state: &DurableState,
    checkpoint: &RemoteDriverCheckpoint,
    kind: AnalysisWorkKind,
    active: bool,
) -> Option<runmat_execution::TaskId> {
    analysis_tasks(control, state, checkpoint, kind, active)
        .into_iter()
        .next()
}

fn committed_pool_tasks(
    control: &DurableControl,
    state: &DurableState,
    checkpoint: &RemoteDriverCheckpoint,
) -> BTreeSet<runmat_execution::TaskId> {
    stored_pool_checkpoint(control, state, checkpoint)
        .map(|(pool, _)| {
            pool.test_snapshot()
                .tasks
                .iter()
                .filter_map(|(id, task)| task.committed.as_ref().map(|_| *id))
                .collect()
        })
        .unwrap_or_default()
}

fn analysis_tasks(
    control: &DurableControl,
    state: &DurableState,
    checkpoint: &RemoteDriverCheckpoint,
    kind: AnalysisWorkKind,
    active: bool,
) -> Vec<runmat_execution::TaskId> {
    use runmat_analysis_execution::AnalysisHostWorkload;
    use runmat_execution::value::{ValuePayload, ValueRefKind};

    let Some((pool, closure)) = stored_pool_checkpoint(control, state, checkpoint) else {
        return Vec::new();
    };
    let mut tasks = Vec::new();
    for (task_id, record) in &pool.test_snapshot().tasks {
        let relevant = if active {
            record.committed.is_none() && record.active_attempt.is_some()
        } else {
            record.committed.is_some()
        };
        if !relevant {
            continue;
        }
        let Some(program) = pool.test_program(*task_id) else {
            continue;
        };
        let Some(ValuePayload::Object(descriptor)) = program.arguments.first() else {
            continue;
        };
        let Some(location) = closure.objects.iter().find(|location| {
            location.reference.id == descriptor.id
                && super::object_transfer::references_share_content_identity(
                    &location.reference,
                    descriptor,
                )
        }) else {
            continue;
        };
        let Some(stored) = state.artifacts.get(&location.artifact_id) else {
            continue;
        };
        let purpose = match location.reference.kind {
            ValueRefKind::DriverObject => EncryptionPurpose::Input,
            ValueRefKind::ResultObject => EncryptionPurpose::Result,
            _ => continue,
        };
        let Ok(bytes) = open_object(&control.run_key, &stored.ciphertext, RUN_ID, purpose) else {
            continue;
        };
        let descriptor_object = ProgramExecutionObject {
            reference: location.reference.clone(),
            bytes: bytes.clone(),
        };
        if descriptor_object.validate().is_err() {
            continue;
        }
        let access = ArtifactAccess {
            authorization_scope: RUN_ID.into(),
            encryption_context: descriptor.encryption_context,
        };
        if AnalysisHostWorkload::from_canonical_bytes(&bytes, access)
            .is_ok_and(|host| host.workload.kind == kind)
        {
            tasks.push(*task_id);
        }
    }
    tasks
}

#[async_trait]
impl DriverControlPlane for DurableControl {
    async fn clock_heartbeat(
        &self,
        authority: &DriverAuthority,
        _ttl_seconds: u64,
        request: runmat_execution_transport_native::control::ClockDriverHeartbeatRequestV1,
    ) -> runmat_execution_transport_native::TransportResult<
        runmat_execution_transport_native::control::ClockDriverHeartbeatResponseV1,
    > {
        let clock = self
            .clock_control
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| unavailable("fixture has no original node clock owner"))?;
        clock.heartbeat(self, authority, request)
    }
    async fn worker_retirement(
        &self,
        authority: &DriverAuthority,
        original: &WorkerRetirementLookup,
    ) -> runmat_execution_transport_native::TransportResult<Option<WorkerTreeRetirementEvidenceV1>>
    {
        if authority.run_id != RUN_ID
            || original.run_id != RUN_ID
            || authority.fencing_token <= original.driver_fencing_token
        {
            return Err(runmat_execution_transport_native::TransportError::StaleAuthority);
        }
        let evidence = self
            .state
            .lock()
            .unwrap()
            .retirements
            .get(&original.allocation_lease_id)
            .cloned();
        if let Some(evidence) = &evidence {
            if evidence.allocation_fencing_token != original.allocation_fencing_token
                || evidence.driver_fencing_token != original.driver_fencing_token
            {
                return Err(runmat_execution_transport_native::TransportError::StaleAuthority);
            }
        }
        Ok(evidence)
    }
    async fn bootstrap(
        &self,
        authority: &DriverAuthority,
    ) -> runmat_execution_transport_native::TransportResult<DriverBootstrap> {
        let mut state = self.state.lock().unwrap();
        let checkpoint_id = state.latest_checkpoint.clone();
        state
            .bootstrap_checkpoints
            .push((authority.fencing_token, checkpoint_id.clone()));
        Ok(DriverBootstrap {
            project_revision: self.project_revision.clone(),
            endpoint_fingerprint: self.endpoint_fingerprint.clone(),
            run_key_envelope: self.run_key_envelope.clone(),
            artifacts: state.initial_artifacts.clone(),
            checkpoint: checkpoint_id
                .as_ref()
                .map(|id| artifact_download(id, state.artifacts.get(id).unwrap())),
            cancellation_requested: false,
            driver_resources: {
                let mut resources = self.resources.clone();
                resources.memory_bytes =
                    runmat_execution::resource::remote_driver_memory_bytes(resources.memory_bytes)
                        .unwrap();
                resources
            },
            driver_accelerator_devices: Vec::new(),
            desired_worker_count: 1,
            worker_resources: self.resources.clone(),
        })
    }

    async fn heartbeat(
        &self,
        authority: &DriverAuthority,
        _ttl_seconds: u64,
    ) -> runmat_execution_transport_native::TransportResult<DriverHeartbeat> {
        let mut state = self.state.lock().unwrap();
        let cancellation_requested = authority.fencing_token == 2
            && self.interruption == ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
            && state.latest_checkpoint.as_ref().is_some_and(|artifact_id| {
                let checkpoint = stored_driver_checkpoint(self, &state, artifact_id);
                let Some(task_id) = self.active_interrupted_task(&state, &checkpoint) else {
                    return false;
                };
                let Some((pool, _)) = stored_pool_checkpoint(self, &state, &checkpoint) else {
                    return false;
                };
                let snapshot = pool.test_snapshot();
                let task = &snapshot.tasks[&task_id];
                task.attempt_count == 2
                    && task.active_attempt.is_some_and(|id| {
                        snapshot
                            .attempts
                            .get(&id)
                            .is_some_and(|attempt| attempt.request.driver_fence == 2)
                    })
            });
        state.cancellation_heartbeat_observed |= cancellation_requested;
        Ok(DriverHeartbeat {
            run_state: "running".into(),
            cancellation_requested,
            expires_at_millis: i64::MAX,
        })
    }

    async fn record_usage(
        &self,
        _authority: &DriverAuthority,
        _source_sequence: u64,
    ) -> runmat_execution_transport_native::TransportResult<bool> {
        Ok(true)
    }

    async fn download(
        &self,
        artifact: &DriverArtifactDownload,
    ) -> runmat_execution_transport_native::TransportResult<Vec<u8>> {
        let state = self.state.lock().unwrap();
        let stored = state
            .artifacts
            .get(&artifact.artifact_id)
            .ok_or_else(|| unavailable("fixture artifact is absent"))?;
        if stored.ciphertext.len() as u64 != artifact.ciphertext_size_bytes
            || ciphertext_digest(&stored.ciphertext) != artifact.ciphertext_digest
        {
            return Err(runmat_execution_transport_native::TransportError::Integrity);
        }
        Ok(stored.ciphertext.clone())
    }

    async fn execution_artifact(
        &self,
        _authority: &DriverAuthority,
        artifact_id: &str,
    ) -> runmat_execution_transport_native::TransportResult<DriverArtifactDownload> {
        let state = self.state.lock().unwrap();
        let artifact = state
            .artifacts
            .get(artifact_id)
            .ok_or_else(|| unavailable("run artifact is absent"))?;
        Ok(artifact_download(artifact_id, artifact))
    }

    async fn store_artifact(
        &self,
        authority: &DriverAuthority,
        artifact: StoreDriverArtifact<'_>,
    ) -> runmat_execution_transport_native::TransportResult<StoredDriverArtifact> {
        if ciphertext_digest(artifact.ciphertext) != artifact.ciphertext_digest {
            return Err(runmat_execution_transport_native::TransportError::Integrity);
        }
        let kind = artifact_kind(artifact.kind);
        let observer_worker_admitted = self
            .clock_control
            .lock()
            .unwrap()
            .as_ref()
            .is_none_or(|clock| clock.worker_admitted());
        let (committed_replacement_checkpoint, interrupted_committed_task) = artifact
            .checkpoint
            .filter(|(_, disposition)| *disposition == CheckpointDisposition::ResumeSafe)
            .and_then(|_| {
                open_object(
                    &self.run_key,
                    artifact.ciphertext,
                    RUN_ID,
                    EncryptionPurpose::Checkpoint,
                )
                .ok()
                .and_then(|bytes| RemoteDriverCheckpoint::from_canonical_bytes(&bytes).ok())
            })
            .map(|checkpoint| match self.interruption {
                ReplacementInterruption::RecoveryClosure => {
                    (checkpoint.closure_catalog.is_some(), None)
                }
                ReplacementInterruption::PoolTaskCommitted => {
                    let state = self.state.lock().unwrap();
                    let previous = state
                        .latest_checkpoint
                        .as_ref()
                        .map(|id| stored_driver_checkpoint(self, &state, id))
                        .map(|prior| committed_pool_tasks(self, &state, &prior))
                        .unwrap_or_default();
                    let newly_committed = committed_pool_tasks(self, &state, &checkpoint)
                        .difference(&previous)
                        .next()
                        .copied();
                    (newly_committed.is_some(), newly_committed)
                }
                ReplacementInterruption::AnalysisCommitted
                | ReplacementInterruption::AnalysisCheckpointAckLost
                | ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore => {
                    let state = self.state.lock().unwrap();
                    let previous = state
                        .latest_checkpoint
                        .as_ref()
                        .map(|id| stored_driver_checkpoint(self, &state, id))
                        .map(|prior| {
                            analysis_tasks(
                                self,
                                &state,
                                &prior,
                                self.interrupted_analysis_kind(),
                                false,
                            )
                        })
                        .unwrap_or_default();
                    let newly_committed = analysis_tasks(
                        self,
                        &state,
                        &checkpoint,
                        self.interrupted_analysis_kind(),
                        false,
                    )
                    .into_iter()
                    .find(|task_id| !previous.contains(task_id));
                    (newly_committed.is_some(), newly_committed)
                }
                ReplacementInterruption::WorkloadAttemptActive
                | ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
                | ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore
                | ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore => {
                    let state = self.state.lock().unwrap();
                    let current_attempt = self
                        .active_interrupted_task(&state, &checkpoint)
                        .and_then(|task_id| {
                            let (pool, _) = stored_pool_checkpoint(self, &state, &checkpoint)?;
                            let snapshot = pool.test_snapshot();
                            let attempt_id = snapshot.tasks[&task_id].active_attempt?;
                            Some(
                                snapshot.attempts[&attempt_id].request.driver_fence
                                    == authority.fencing_token,
                            )
                        })
                        .unwrap_or(false);
                    (current_attempt, None)
                }
                ReplacementInterruption::ReplayableExecutionStart => (false, None),
            })
            .unwrap_or((false, None));
        let (stored, interrupt_driver) = {
            let mut state = self.state.lock().unwrap();
            if authority.fencing_token == 1
                && committed_replacement_checkpoint
                && self.interruption
                    == ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore
                && !state.driver_interruption_triggered
            {
                state.driver_interruption_triggered = true;
                state.interrupted_committed_task = interrupted_committed_task;
                return Err(unavailable(
                    "injected checkpoint rejection before persistence",
                ));
            }
            state.next_artifact += 1;
            let artifact_id = format!("result-{}", state.next_artifact);
            state.artifacts.insert(
                artifact_id.clone(),
                StoredArtifact {
                    kind: kind.into(),
                    ciphertext: artifact.ciphertext.to_vec(),
                },
            );
            if artifact.checkpoint.is_some() {
                state.latest_checkpoint = Some(artifact_id.clone());
            }
            let interrupt_driver = committed_replacement_checkpoint
                && observer_worker_admitted
                && ((authority.fencing_token == 1 && !state.driver_interruption_triggered)
                    || (authority.fencing_token == 2
                        && self.interruption
                            == ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore
                        && !state.second_driver_interruption_triggered));
            if interrupt_driver {
                if authority.fencing_token == 1 {
                    state.driver_interruption_triggered = true;
                    state.interrupted_committed_task = interrupted_committed_task;
                } else {
                    state.second_driver_interruption_triggered = true;
                }
            }
            (
                StoredDriverArtifact {
                    artifact_id,
                    kind: kind.into(),
                    ciphertext_size_bytes: artifact.ciphertext.len() as u64,
                },
                interrupt_driver,
            )
        };
        if artifact.checkpoint.is_some() {
            let clock = self.clock_control.lock().unwrap().clone();
            if let Some(clock) = clock {
                clock.before_checkpoint_ack().await;
            }
        }
        if interrupt_driver {
            self.recovery_committed.notify_waiters();
            if self.interruption == ReplacementInterruption::AnalysisCheckpointAckLost {
                return Err(unavailable("injected checkpoint acknowledgement loss"));
            }
            return std::future::pending().await;
        }
        Ok(stored)
    }

    async fn transition(
        &self,
        authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> runmat_execution_transport_native::TransportResult<()> {
        if transition.target == DriverRunTarget::Succeeded {
            if let Some(clock) = self.clock_control.lock().unwrap().as_ref() {
                clock.assert_released_before_success();
            }
        }
        self.state.lock().unwrap().transitions.push((
            authority.fencing_token,
            transition.target,
            transition.result_artifact_id,
        ));
        Ok(())
    }

    async fn resize_workers(
        &self,
        authority: &DriverAuthority,
        expected_generation: u64,
        desired_workers: u32,
        resources: ResourceRequest,
    ) -> runmat_execution_transport_native::TransportResult<DriverWorkerPool> {
        if resources != self.resources
            || !matches!((expected_generation, desired_workers), (0, 1) | (1, 0))
        {
            return Err(runmat_execution_transport_native::TransportError::Integrity);
        }
        Ok(DriverWorkerPool {
            generation: expected_generation + 1,
            desired_workers,
            resources,
            workers: if desired_workers == 0 {
                Vec::new()
            } else {
                vec![self.workers.get(&authority.fencing_token).unwrap().clone()]
            },
        })
    }

    async fn list_workers(
        &self,
        authority: &DriverAuthority,
    ) -> runmat_execution_transport_native::TransportResult<DriverWorkerPool> {
        Ok(DriverWorkerPool {
            generation: 1,
            desired_workers: 1,
            resources: self.resources.clone(),
            workers: vec![self.workers.get(&authority.fencing_token).unwrap().clone()],
        })
    }

    async fn authorize_worker(
        &self,
        authority: &DriverAuthority,
        allocation_lease_id: &str,
        endpoint_fingerprint: &str,
        run_key_envelope: &[u8],
        _ready_v1: bool,
    ) -> runmat_execution_transport_native::TransportResult<()> {
        let worker = self.workers.get(&authority.fencing_token).unwrap();
        let evidence = worker.endpoint_identity.as_ref().unwrap();
        if allocation_lease_id != worker.allocation_lease_id
            || endpoint_fingerprint != evidence.recipient.fingerprint
            || run_key_envelope.is_empty()
        {
            return Err(runmat_execution_transport_native::TransportError::Integrity);
        }
        let interrupt_driver = {
            let mut state = self.state.lock().unwrap();
            state
                .worker_authorizations
                .push((authority.fencing_token, allocation_lease_id.into()));
            let interrupt_driver = self.interruption
                == ReplacementInterruption::ReplayableExecutionStart
                && authority.fencing_token == 1
                && !state.driver_interruption_triggered;
            if interrupt_driver {
                state.driver_interruption_triggered = true;
            }
            interrupt_driver
        };
        if interrupt_driver {
            self.recovery_committed.notify_waiters();
            return std::future::pending().await;
        }
        Ok(())
    }
}

#[test]
fn exact_meshing_root_preserves_committed_pool_task_after_encrypted_driver_replacement() {
    assert!(run_actual_driver_replacement(ExactStudyFamily::MeshingRoot).is_none());
}

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/remote_execution; features: occt-native;
//   coverage: partial;
//   boundary: Exact meshing admission retries and cancels after an actual encrypted driver
//   replacement.
#[test]
fn exact_meshing_admission_retries_and_cancels_after_encrypted_driver_replacement() {
    assert!(run_actual_driver_replacement_with_interruption(
        ExactStudyFamily::MeshingRoot,
        ReplacementInterruption::WorkloadAttemptActive,
    )
    .is_none());
    assert!(run_actual_driver_replacement_with_interruption(
        ExactStudyFamily::MeshingRoot,
        ReplacementInterruption::WorkloadAttemptCancelledAfterRestore,
    )
    .is_none());
}

#[test]
fn exact_meshing_admission_deadline_expires_before_replacement_dispatch() {
    assert!(run_actual_driver_replacement_with_interruption(
        ExactStudyFamily::MeshingRoot,
        ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore,
    )
    .is_none());
}

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/remote_execution; features: occt-native;
//   coverage: partial;
//   boundary: The retry budget is exhausted after two driver losses; worker-process loss and
//   transfer resumption are covered by the process-conformance target.
#[test]
fn exact_meshing_admission_exhausts_retry_budget_after_two_driver_losses() {
    assert!(run_actual_driver_replacement_with_interruption(
        ExactStudyFamily::MeshingRoot,
        ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore,
    )
    .is_none());
}

#[test]
fn actual_driver_restarts_from_server_checkpoint_on_a_different_worker() {
    run_actual_driver_replacement(ExactStudyFamily::StructuralTransient);
}

#[test]
fn response_spectrum_driver_replacement_replays_on_a_different_worker() {
    run_actual_driver_replacement(ExactStudyFamily::ModalResponseSpectrum);
}

#[test]
fn random_response_driver_replacement_replays_on_a_different_worker() {
    run_actual_driver_replacement(ExactStudyFamily::ModalRandomResponse);
}

#[test]
fn authored_thermal_outer_root_matrix_replays_after_preexecution_driver_replacement() {
    // Thermal interruption occurs before execution starts. This is not a claim of restart
    // from a completed heat/preload/basis phase; those recovery protocols are distinct.
    let results = thermal::Case::matrix()
        .map(|case| {
            run_actual_driver_replacement(ExactStudyFamily::ThermalPrestress(case)).unwrap()
        })
        .collect::<Vec<_>>();
    for (index, result) in results.iter().enumerate() {
        result.reject_foreign_thermal(&results[(index + 1) % results.len()]);
    }
}

#[test]
fn authored_thermal_prefix_survives_driver_loss_after_its_commit() {
    for case in thermal::Case::matrix() {
        let result = run_actual_driver_replacement_with_interruption(
            ExactStudyFamily::ThermalPrestress(case),
            ReplacementInterruption::AnalysisCommitted,
        );
        assert!(result.is_some());
    }
}

#[test]
fn authored_thermal_prefix_survives_lost_checkpoint_acknowledgement() {
    for case in thermal::Case::matrix() {
        let result = run_actual_driver_replacement_with_interruption(
            ExactStudyFamily::ThermalPrestress(case),
            ReplacementInterruption::AnalysisCheckpointAckLost,
        );
        assert!(result.is_some());
    }
}

#[test]
fn rejected_thermal_checkpoint_cannot_be_restored_as_a_committed_prefix() {
    for case in thermal::Case::matrix() {
        let result = run_actual_driver_replacement_with_interruption(
            ExactStudyFamily::ThermalPrestress(case),
            ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore,
        );
        assert!(result.is_some());
    }
}

#[test]
fn committed_structural_and_modal_work_survives_checkpoint_acknowledgement_loss() {
    for family in nonthermal_analysis_replacement_families() {
        run_actual_driver_replacement_with_interruption(
            family,
            ReplacementInterruption::AnalysisCheckpointAckLost,
        );
    }
}

#[test]
fn rejected_structural_and_modal_checkpoint_does_not_grant_commit_authority() {
    for family in nonthermal_analysis_replacement_families() {
        run_actual_driver_replacement_with_interruption(
            family,
            ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore,
        );
    }
}

fn nonthermal_analysis_replacement_families() -> impl Iterator<Item = ExactStudyFamily> {
    [
        ExactStudyFamily::StructuralTransient,
        ExactStudyFamily::ModalResponseSpectrum,
        ExactStudyFamily::ModalRandomResponse,
    ]
    .into_iter()
}

fn analysis_replacement_families() -> impl Iterator<Item = ExactStudyFamily> {
    nonthermal_analysis_replacement_families()
        .chain(thermal::Case::matrix().map(ExactStudyFamily::ThermalPrestress))
}

#[test]
fn active_analysis_attempt_retries_after_actual_driver_replacement() {
    for family in analysis_replacement_families() {
        let result = run_actual_driver_replacement_with_interruption(
            family,
            ReplacementInterruption::WorkloadAttemptActive,
        );
        assert_eq!(
            result.is_some(),
            matches!(family, ExactStudyFamily::ThermalPrestress(_))
        );
    }
}

#[test]
fn restored_analysis_retry_observes_cancellation_before_publication() {
    for family in analysis_replacement_families() {
        let result = run_actual_driver_replacement_with_interruption(
            family,
            ReplacementInterruption::WorkloadAttemptCancelledAfterRestore,
        );
        assert!(result.is_none());
    }
}

fn run_actual_driver_replacement(family: ExactStudyFamily) -> Option<thermal::ColdReplay> {
    run_actual_driver_replacement_with_interruption(family, family.interruption())
}

fn run_actual_driver_replacement_with_interruption(
    family: ExactStudyFamily,
    interruption: ReplacementInterruption,
) -> Option<thermal::ColdReplay> {
    // Production driver startup requires its original node-clock parent. Every
    // actual replacement fixture uses that same signed wall authority, including
    // work that began before the first driver was lost.
    run_actual_driver_replacement_with_observer(family, interruption, true)
}

fn run_actual_driver_replacement_with_observer(
    family: ExactStudyFamily,
    interruption: ReplacementInterruption,
    observe: bool,
) -> Option<thermal::ColdReplay> {
    enter_worker_subprocess_if_requested();
    let test_name = std::thread::current()
        .name()
        .expect("named Rust test thread")
        .to_string();
    std::thread::Builder::new()
        .name(format!("actual-driver-replacement-{family:?}"))
        .stack_size(ACTUAL_DRIVER_TEST_STACK_BYTES)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actual-driver replacement runtime")
                .block_on(actual_driver_replacement_case(
                    family,
                    interruption,
                    &test_name,
                    observe,
                ))
        })
        .expect("actual-driver replacement test thread")
        .join()
        .expect("actual-driver replacement test thread panicked")
}

fn assert_exhausted_replacement(
    control: &DurableControl,
    first_checkpoint_id: &str,
    second_checkpoint_id: &str,
    active_before: (
        runmat_execution::TaskId,
        runmat_execution_runner::DriverSnapshot,
        runmat_execution::identity::ValueId,
    ),
) {
    use runmat_execution::state::{AttemptState, TaskState};
    use runmat_execution::task::RetryPolicy;
    use runmat_execution_runner::driver::DriverEventKind;

    let (task_id, before, _) = active_before;
    let state = control.state.lock().unwrap();
    let first = stored_driver_checkpoint(control, &state, first_checkpoint_id);
    let second = stored_driver_checkpoint(control, &state, second_checkpoint_id);
    let final_id = state
        .latest_checkpoint
        .as_ref()
        .expect("terminal checkpoint");
    let terminal = stored_driver_checkpoint(control, &state, final_id);
    assert!(first.sequence < second.sequence && second.sequence < terminal.sequence);
    let (second_pool, _) = stored_pool_checkpoint(control, &state, &second)
        .expect("second active scheduler authority");
    let second_snapshot = second_pool.test_snapshot();
    let second_task = &second_snapshot.tasks[&task_id];
    assert_eq!(second_task.submission, before.tasks[&task_id].submission);
    assert_eq!(second_task.attempt_count, 2);
    let second_attempt = second_task.active_attempt.expect("second active attempt");
    assert_eq!(
        second_snapshot.attempts[&second_attempt]
            .request
            .driver_fence,
        2
    );
    let first_attempt = before.tasks[&task_id]
        .active_attempt
        .expect("first active attempt");
    assert_eq!(
        second_snapshot.attempts[&first_attempt].state,
        AttemptState::Lost
    );

    let (final_pool, _) =
        stored_pool_checkpoint(control, &state, &terminal).expect("terminal scheduler authority");
    let after = final_pool.test_snapshot();
    let exhausted = &after.tasks[&task_id];
    assert_eq!(exhausted.submission, before.tasks[&task_id].submission);
    assert_eq!(
        exhausted.submission.request.retry,
        RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 }
    );
    assert_eq!(exhausted.state, TaskState::Failed);
    assert_eq!(exhausted.attempt_count, 2);
    assert!(exhausted.active_attempt.is_none() && exhausted.committed.is_none());
    assert_eq!(after.tasks.len(), before.tasks.len());
    assert_eq!(after.attempts.len(), 2);
    for attempt_id in [first_attempt, second_attempt] {
        assert_eq!(after.attempts[&attempt_id].state, AttemptState::Lost);
    }
    assert!(after.events.iter().any(|event| {
        event.driver_fence == 3
            && matches!(
                &event.kind,
                DriverEventKind::TaskStateChanged {
                    task_id: changed,
                    state: TaskState::Failed
                } if *changed == task_id
            )
    }));
    assert_eq!(
        state.bootstrap_checkpoints,
        vec![
            (1, None),
            (2, Some(first_checkpoint_id.into())),
            (3, Some(second_checkpoint_id.into()))
        ]
    );
    assert_eq!(state.worker_authorizations.len(), 3);
    assert_eq!(
        state
            .transitions
            .iter()
            .filter(|(fence, target, _)| *fence == 3 && *target == DriverRunTarget::Failed)
            .count(),
        1
    );
    assert!(state.transitions.iter().all(|(_, target, _)| {
        !matches!(
            target,
            DriverRunTarget::Succeeded | DriverRunTarget::Indeterminate
        )
    }));
}

async fn actual_driver_replacement_case(
    family: ExactStudyFamily,
    interruption: ReplacementInterruption,
    test_name: &str,
    observe: bool,
) -> Option<thermal::ColdReplay> {
    let mut fixture =
        if interruption == ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore {
            // Long enough for the first authenticated active-attempt checkpoint;
            // short enough to expire during the deliberate driver outage below.
            exact_study_fixture_with_wall_time(family, 15_000)
        } else {
            exact_study_fixture(family)
        };
    fixture.control.interruption = interruption;
    let driver_material = driver_identity_material();
    let directory = tempfile::tempdir().unwrap();
    let identity_path = directory.path().join("driver-endpoint.json");
    std::fs::write(
        &identity_path,
        serde_json::to_vec(&driver_material).unwrap(),
    )
    .unwrap();
    let control = Arc::new(fixture.control);
    let clock_domain_nonce: [u8; 16] = loop {
        let nonce: [u8; 16] = rand::random();
        if nonce.iter().any(|value| *value != 0) {
            break nonce;
        }
    };
    let first_journal = worker_journal(
        directory.path(),
        control.workers.get(&1).unwrap(),
        1,
        clock_domain_nonce,
    );
    let clock = observe
        .then(|| observer::ClockControl::new(directory.path(), first_journal.directory.clone()));
    let _clock_owner = clock.as_ref().map(|clock| {
        *control.clock_control.lock().unwrap() = Some(Arc::clone(clock));
        clock.spawn_owner()
    });
    let first_relay = RelayServer::bind().await;
    let first_authority = driver_authority(1, first_relay.http_base());
    let mut first_worker = spawn_worker_process(
        test_name,
        first_relay.ws_url("worker"),
        control.workers.get(&1).unwrap(),
        &fixture.inventory,
        &fixture.run_key,
        matches!(
            interruption,
            ReplacementInterruption::WorkloadAttemptActive
                | ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
                | ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore
                | ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore
        )
        .then(|| super::worker_execution::HeldWorkloadAttempt {
            target: control.interrupted_target(),
            input_object_id: None,
            until: super::worker_execution::HeldAttemptUntil::UntilWorkerLoss,
        }),
        first_journal.clone(),
    )
    .await;
    let first_pid = first_worker.id().expect("first OS worker PID");
    let first_driver = super::driver::run_remote_driver(
        RemoteDriverConfig {
            authority: first_authority,
            endpoint_identity_file: identity_path.clone(),
            ready_v1: false,
            clock_parent: clock.as_ref().map(|clock| clock.parent(1)),
        },
        control.clone(),
    );
    if matches!(
        interruption,
        ReplacementInterruption::AnalysisCheckpointAckLost
            | ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore
    ) {
        let generation = drive_generation(
            first_relay.forward_pair(),
            async {
                first_worker
                    .wait_and_retire_tree()
                    .await
                    .map(|_| ())
                    .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))
            },
            first_driver,
        )
        .await;
        let expected = match interruption {
            ReplacementInterruption::AnalysisCheckpointAckLost => {
                "injected checkpoint acknowledgement loss"
            }
            ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore => {
                "injected checkpoint rejection before persistence"
            }
            _ => unreachable!(),
        };
        match generation {
            Err(error) => {
                let injected = control.state.lock().unwrap().driver_interruption_triggered;
                assert!(
                    injected
                        && (error.to_string().contains(expected)
                            || error.to_string().contains(
                                "remote checkpoint acknowledgement is uncertain; replace the driver"
                            )),
                    "unexpected first-driver failure: {error}"
                );
            }
            Ok(()) => {
                let state = control.state.lock().unwrap();
                assert!(
                    state.transitions.iter().any(|(fence, target, _)| {
                        *fence == 1 && *target == DriverRunTarget::Failed
                    }) && state.transitions.iter().all(|(fence, target, _)| {
                        *fence != 1 || *target != DriverRunTarget::Succeeded
                    }),
                    "first driver did not fail after {expected}: {family:?}, transitions={:?}",
                    state.transitions
                );
            }
        }
    } else {
        interrupt_generation_after_commit(
            first_relay.forward_pair(),
            async {
                first_worker
                    .wait_and_retire_tree()
                    .await
                    .map(|_| ())
                    .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))
            },
            first_driver,
            control.recovery_committed.notified(),
        )
        .await;
    }
    first_worker
        .terminate_tree()
        .await
        .expect("first worker process tree retired");
    if let Some(clock) = &clock {
        if interruption == ReplacementInterruption::ReplayableExecutionStart {
            clock.assert_preexecution_worker_not_admitted();
        } else {
            clock.assert_worker_started_after_barrier(&fixture.run_key);
        }
    }
    control.record_retired_worker(
        1,
        first_pid,
        &first_journal,
        runmat_process_host::clock::observe_monotonic().unwrap(),
    );
    if observe {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    let checkpoint_after_loss = {
        let state = control.state.lock().unwrap();
        assert!(state.driver_interruption_triggered);
        state.latest_checkpoint.clone().unwrap()
    };
    if interruption == ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore {
        let state = control.state.lock().unwrap();
        let checkpoint = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        let rejected_task = state
            .interrupted_committed_task
            .expect("rejected checkpoint must identify its new commit");
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint)
            .expect("prior checkpoint retains scheduler authority");
        assert!(
            pool.test_snapshot()
                .tasks
                .get(&rejected_task)
                .is_none_or(|task| task.committed.is_none()),
            "an unrecorded checkpoint cannot grant its new task committed-prefix authority"
        );
    }
    let committed_before = if matches!(
        interruption,
        ReplacementInterruption::PoolTaskCommitted
            | ReplacementInterruption::AnalysisCommitted
            | ReplacementInterruption::AnalysisCheckpointAckLost
    ) {
        let state = control.state.lock().unwrap();
        let checkpoint = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        let task_id = state
            .interrupted_committed_task
            .expect("driver loss must follow an authenticated new pool task commit");
        assert!(committed_pool_tasks(&control, &state, &checkpoint).contains(&task_id));
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint)
            .expect("committed pool checkpoint");
        let programs = pool
            .test_snapshot()
            .tasks
            .iter()
            .filter(|(_, task)| task.committed.is_some())
            .map(|(id, _)| {
                (
                    *id,
                    pool.test_program(*id).expect("committed program").clone(),
                )
            })
            .collect::<Vec<_>>();
        let walls = pool
            .test_snapshot()
            .tasks
            .iter()
            .filter(|(_, task)| task.committed.is_some())
            .filter_map(|(id, _)| {
                pool.test_completed_task_wall_millis(*id)
                    .map(|wall| (*id, wall))
            })
            .collect::<BTreeMap<_, _>>();
        Some((task_id, pool.test_snapshot().clone(), programs, walls))
    } else {
        None
    };
    let active_task_before = if matches!(
        interruption,
        ReplacementInterruption::WorkloadAttemptActive
            | ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
            | ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore
            | ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore
    ) {
        let state = control.state.lock().unwrap();
        let checkpoint = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        let task_id = control
            .active_interrupted_task(&state, &checkpoint)
            .expect("driver loss must follow an authenticated active workload attempt");
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint).unwrap();
        let input_object_id = match pool
            .test_program(task_id)
            .expect("active workload program")
            .arguments
            .first()
        {
            Some(runmat_execution::value::ValuePayload::Object(reference)) => reference.id,
            _ => panic!("active workload input is not an object"),
        };
        Some((task_id, pool.test_snapshot().clone(), input_object_id))
    } else {
        None
    };
    if interruption == ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore {
        let (task_id, before, _) = active_task_before
            .as_ref()
            .expect("active workload checkpoint");
        let task = &before.tasks[task_id];
        let deadline = task
            .submission
            .request
            .deadline_unix_millis
            .expect("admitted absolute deadline");
        tokio::time::sleep(std::time::Duration::from_millis(
            deadline.saturating_sub(now_millis()).saturating_add(100),
        ))
        .await;
    }
    let second_journal = worker_journal(
        directory.path(),
        control.workers.get(&2).unwrap(),
        2,
        clock_domain_nonce,
    );
    let second_relay = RelayServer::bind().await;
    let second_authority = driver_authority(2, second_relay.http_base());
    let mut second_worker = spawn_worker_process(
        test_name,
        second_relay.ws_url("worker"),
        control.workers.get(&2).unwrap(),
        &fixture.inventory,
        &fixture.run_key,
        matches!(
            interruption,
            ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
                | ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore
        )
        .then(|| super::worker_execution::HeldWorkloadAttempt {
            target: control.interrupted_target(),
            input_object_id: active_task_before
                .as_ref()
                .map(|(_, _, object_id)| *object_id),
            until: if interruption == ReplacementInterruption::WorkloadAttemptCancelledAfterRestore
            {
                super::worker_execution::HeldAttemptUntil::UntilCancellation
            } else {
                super::worker_execution::HeldAttemptUntil::UntilWorkerLoss
            },
        }),
        second_journal.clone(),
    )
    .await;
    let second_pid = second_worker.id().expect("second OS worker PID");
    let second_driver = super::driver::run_remote_driver(
        RemoteDriverConfig {
            authority: second_authority,
            endpoint_identity_file: identity_path.clone(),
            ready_v1: false,
            clock_parent: clock.as_ref().map(|clock| clock.parent(2)),
        },
        control.clone(),
    );
    if interruption == ReplacementInterruption::WorkloadRetryBudgetExhaustedAfterRestore {
        interrupt_generation_after_commit(
            second_relay.forward_pair(),
            async {
                second_worker
                    .wait_and_retire_tree()
                    .await
                    .map(|_| ())
                    .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))
            },
            second_driver,
            control.recovery_committed.notified(),
        )
        .await;
        second_worker
            .terminate_tree()
            .await
            .expect("second worker process tree retired");
        control.record_retired_worker(
            2,
            second_pid,
            &second_journal,
            runmat_process_host::clock::observe_monotonic().unwrap(),
        );
        assert!(
            control
                .state
                .lock()
                .unwrap()
                .second_driver_interruption_triggered
        );
        let second_checkpoint_id = control
            .state
            .lock()
            .unwrap()
            .latest_checkpoint
            .clone()
            .expect("second active-attempt checkpoint");
        let third_relay = RelayServer::bind().await;
        let third_journal = worker_journal(
            directory.path(),
            control.workers.get(&3).unwrap(),
            3,
            clock_domain_nonce,
        );
        let mut third_worker = spawn_worker_process(
            test_name,
            third_relay.ws_url("worker"),
            control.workers.get(&3).unwrap(),
            &fixture.inventory,
            &fixture.run_key,
            None,
            third_journal,
        )
        .await;
        let third_driver = super::driver::run_remote_driver(
            RemoteDriverConfig {
                authority: driver_authority(3, third_relay.http_base()),
                endpoint_identity_file: identity_path,
                ready_v1: false,
                clock_parent: clock.as_ref().map(|clock| clock.parent(3)),
            },
            control.clone(),
        );
        drive_generation(
            third_relay.forward_pair(),
            async {
                third_worker
                    .wait_and_retire_tree()
                    .await
                    .map(|_| ())
                    .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))
            },
            third_driver,
        )
        .await
        .expect("exhausted third driver must publish a terminal failure");
        third_worker
            .terminate_tree()
            .await
            .expect("third worker process tree retired");
        assert_exhausted_replacement(
            &control,
            &checkpoint_after_loss,
            &second_checkpoint_id,
            active_task_before.expect("first active-attempt checkpoint"),
        );
        return None;
    }
    let driver_result = drive_generation(
        second_relay.forward_pair(),
        async {
            second_worker
                .wait_and_retire_tree()
                .await
                .map(|_| ())
                .map_err(|error| crate::NativeExecutionError::Protocol(error.to_string()))
        },
        second_driver,
    )
    .await;
    second_worker
        .terminate_tree()
        .await
        .expect("second worker process tree retired");
    if let Err(error) = driver_result {
        let state = control.state.lock().unwrap();
        let checkpoint = state.latest_checkpoint.as_ref().map(|id| {
            let checkpoint = stored_driver_checkpoint(&control, &state, id);
            let decoded_active = control.active_interrupted_task(&state, &checkpoint);
            let tasks = stored_pool_checkpoint(&control, &state, &checkpoint).map(|(pool, _)| {
                let snapshot = pool.test_snapshot();
                (
                    snapshot
                        .tasks
                        .iter()
                        .map(|(id, task)| {
                            (
                                *id,
                                task.state,
                                task.attempt_count,
                                task.active_attempt,
                                active_task_before
                                    .as_ref()
                                    .is_some_and(|(target, _, _)| id == target),
                            )
                        })
                        .collect::<Vec<_>>(),
                    snapshot
                        .attempts
                        .iter()
                        .map(|(id, attempt)| {
                            (
                                *id,
                                attempt.request.task_id,
                                attempt.state,
                                attempt.request.driver_fence,
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            });
            (checkpoint.sequence, decoded_active, tasks)
        });
        panic!(
            "replacement failed for {family:?} {interruption:?}: {error}; first checkpoint={checkpoint_after_loss}, cancellation heartbeat={}, worker authorizations={:?}, checkpoint={checkpoint:?}",
            state.cancellation_heartbeat_observed,
            state.worker_authorizations,
        );
    }

    if interruption == ReplacementInterruption::WorkloadAttemptCancelledAfterRestore {
        use runmat_execution::state::{AttemptState, TaskState};
        let state = control.state.lock().unwrap();
        assert!(state.cancellation_heartbeat_observed);
        assert_eq!(
            state.bootstrap_checkpoints,
            vec![(1, None), (2, Some(checkpoint_after_loss.clone()))]
        );
        assert_eq!(
            state
                .transitions
                .iter()
                .filter(|(fence, target, result)| {
                    *fence == 2 && *target == DriverRunTarget::Cancelled && result.is_none()
                })
                .count(),
            1
        );
        assert!(state.transitions.iter().all(|(_, target, _)| {
            !matches!(
                target,
                DriverRunTarget::Succeeded | DriverRunTarget::Indeterminate
            )
        }));
        assert_eq!(state.worker_authorizations.len(), 2);
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("terminal checkpoint");
        let checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        let before_checkpoint = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        assert!(checkpoint.sequence > before_checkpoint.sequence);
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint).unwrap();
        let (task_id, before, _) = active_task_before.expect("active workload checkpoint");
        let original = &before.tasks[&task_id];
        let after = pool.test_snapshot();
        let cancelled = &after.tasks[&task_id];
        let first_attempt = original.active_attempt.expect("first live attempt");
        assert_eq!(cancelled.submission, original.submission);
        assert_eq!(cancelled.attempt_count, 2);
        assert_eq!(cancelled.state, TaskState::Cancelled);
        assert!(cancelled.committed.is_none());
        assert_eq!(after.attempts[&first_attempt].state, AttemptState::Lost);
        assert_eq!(
            after
                .attempts
                .values()
                .filter(|attempt| attempt.request.task_id == task_id)
                .count(),
            2
        );
        assert!(
            after.attempts.values().any(|attempt| {
                attempt.request.task_id == task_id
                    && attempt.request.driver_fence == 2
                    && attempt.state == AttemptState::Cancelled
            }),
            "restored target attempt states: {:?}",
            after
                .attempts
                .iter()
                .filter(|(_, attempt)| attempt.request.task_id == task_id)
                .map(|(id, attempt)| (id, attempt.state, attempt.cancellation_requested_at))
                .collect::<Vec<_>>()
        );
        return None;
    }

    if interruption == ReplacementInterruption::WorkloadDeadlineExpiredAfterRestore {
        use runmat_execution::state::{AttemptState, TaskState};
        use runmat_execution_runner::driver::DriverEventKind;

        let state = control.state.lock().unwrap();
        let (task_id, before, _) = active_task_before.expect("active workload checkpoint");
        let prior_id = &checkpoint_after_loss;
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("terminal checkpoint");
        let prior = stored_driver_checkpoint(&control, &state, prior_id);
        let final_checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        assert!(final_checkpoint.sequence > prior.sequence);
        let (pool, _) = stored_pool_checkpoint(&control, &state, &final_checkpoint)
            .expect("terminal scheduler authority");
        let after = pool.test_snapshot();
        let original = &before.tasks[&task_id];
        let expired = &after.tasks[&task_id];
        assert_eq!(expired.submission, original.submission);
        assert_eq!(expired.state, TaskState::Cancelled);
        assert_eq!(expired.attempt_count, 1);
        assert!(expired.committed.is_none());
        let first_attempt = original.active_attempt.expect("original active attempt");
        assert_eq!(after.attempts[&first_attempt].state, AttemptState::Lost);
        assert_eq!(after.attempts.len(), before.attempts.len());
        assert!(after.events.iter().any(|event| {
            matches!(&event.kind, DriverEventKind::DeadlineExpired { task_id: expired_id }
                if *expired_id == task_id)
        }));
        assert!(state
            .transitions
            .iter()
            .any(|(fence, target, _)| { *fence == 2 && *target == DriverRunTarget::Cancelled }));
        assert!(state.transitions.iter().all(|(_, target, _)| {
            !matches!(
                target,
                DriverRunTarget::Succeeded | DriverRunTarget::Indeterminate
            )
        }));
        return None;
    }

    let (result_artifact, transitions, authorizations, bootstraps) = {
        let state = control.state.lock().unwrap();
        let result = state
            .transitions
            .iter()
            .find_map(|(fence, target, result)| {
                (*fence == 2 && *target == DriverRunTarget::Succeeded)
                    .then(|| result.clone())
                    .flatten()
            })
            .unwrap_or_else(|| {
                let diagnostics = state
                    .artifacts
                    .values()
                    .filter(|artifact| artifact.kind == artifact_kind(DriverArtifactKind::Diagnostic))
                    .map(|artifact| {
                        open_object(
                            &control.run_key,
                            &artifact.ciphertext,
                            RUN_ID,
                            EncryptionPurpose::DetailedEvent,
                        )
                        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                        .unwrap_or_else(|error| format!("diagnostic decryption failed: {error}"))
                    })
                    .collect::<Vec<_>>();
                panic!(
                    "replacement did not publish success: transitions={:?}; diagnostics={diagnostics:?}",
                    state.transitions
                );
            });
        (
            result,
            state.transitions.clone(),
            state.worker_authorizations.clone(),
            state.bootstrap_checkpoints.clone(),
        )
    };
    if let Some((task_id, before, prefix_programs, prefix_walls)) = committed_before {
        let state = control.state.lock().unwrap();
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("final pool checkpoint");
        let checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint)
            .expect("final scheduler authority is published");
        let checkpoint_at_loss = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        assert!(
            checkpoint.sequence > checkpoint_at_loss.sequence,
            "replacement must advance beyond the committed analysis checkpoint"
        );
        if interruption == ReplacementInterruption::AnalysisCheckpointAckLost {
            let mut sequences = BTreeSet::new();
            for (artifact_id, artifact) in &state.artifacts {
                if artifact.kind == artifact_kind(DriverArtifactKind::Checkpoint) {
                    let stored = stored_driver_checkpoint(&control, &state, artifact_id);
                    assert!(
                        sequences.insert(stored.sequence),
                        "a checkpoint sequence was reused after an ambiguous acknowledgement"
                    );
                }
            }
        }
        let after = pool.test_snapshot();
        assert!(before
            .tasks
            .get(&task_id)
            .is_some_and(|task| task.committed.is_some()));
        for (committed_id, committed) in before
            .tasks
            .iter()
            .filter(|(_, task)| task.committed.is_some())
        {
            if let Some(original_wall) = prefix_walls.get(committed_id) {
                assert_eq!(
                    pool.test_completed_task_wall_millis(*committed_id),
                    Some(*original_wall),
                    "committed prefix lost its original measured wall on cached replay"
                );
            }
            assert_eq!(
                after.tasks.get(committed_id),
                Some(committed),
                "a committed prefix task changed identity, commit or accounting"
            );
            for (attempt_id, attempt) in &before.attempts {
                if attempt.request.task_id == *committed_id {
                    assert_eq!(
                        after.attempts.get(attempt_id),
                        Some(attempt),
                        "a committed prefix attempt was redispatched or altered"
                    );
                }
            }
        }
        for (new_id, _) in after
            .tasks
            .iter()
            .filter(|(id, _)| !before.tasks.contains_key(id))
        {
            let new_program = pool.test_program(*new_id).expect("final program");
            assert!(
                prefix_programs
                    .iter()
                    .all(|(_, prefix_program)| prefix_program != new_program),
                "replacement redispatched a committed prefix under a different task id"
            );
        }
        if matches!(family, ExactStudyFamily::ThermalPrestress(_)) {
            assert!(
                after.tasks.iter().any(|(id, task)| {
                    *id != task_id
                        && task.committed.is_some()
                        && before
                            .tasks
                            .get(id)
                            .is_none_or(|previous| previous.committed.is_none())
                }),
                "replacement must complete downstream work, not replay an already completed study"
            );
        }
    }
    if let Some((task_id, before, _)) = active_task_before {
        use runmat_execution::state::AttemptState;
        let state = control.state.lock().unwrap();
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("final pool checkpoint");
        let checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint).unwrap();
        let after = pool.test_snapshot();
        let original = &before.tasks[&task_id];
        let recovered = &after.tasks[&task_id];
        let old_attempt = original.active_attempt.expect("in-flight workload attempt");
        assert_eq!(recovered.submission, original.submission);
        assert_eq!(original.attempt_count, 1);
        assert_eq!(recovered.attempt_count, 2);
        assert_eq!(recovered.submission.request.id, task_id);
        assert_eq!(
            after.attempts[&old_attempt].request,
            before.attempts[&old_attempt].request
        );
        assert_eq!(after.attempts[&old_attempt].state, AttemptState::Lost);
        let committed = recovered
            .committed
            .as_ref()
            .expect("replacement workload commit");
        assert_ne!(committed.attempt_id, old_attempt);
        assert_eq!(committed.driver_fence, 2);
        assert_eq!(
            after
                .attempts
                .values()
                .filter(|attempt| attempt.request.task_id == task_id)
                .count(),
            2,
        );
        assert!(after.tasks.iter().all(|(id, _)| {
            *id == task_id
                || before.tasks.contains_key(id)
                || pool.test_program(*id) != pool.test_program(task_id)
        }));
    }
    if interruption == ReplacementInterruption::AnalysisCheckpointRejectedBeforeStore {
        let state = control.state.lock().unwrap();
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("final pool checkpoint");
        let checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        assert!(
            analysis_task(
                &control,
                &state,
                &checkpoint,
                control.interrupted_analysis_kind(),
                false
            )
            .is_some(),
            "replacement must actually complete the analysis task"
        );
    }
    assert_eq!(
        bootstraps,
        vec![(1, None), (2, Some(checkpoint_after_loss))]
    );
    assert!(transitions
        .iter()
        .any(|(fence, target, _)| *fence == 1 && *target == DriverRunTarget::Running));
    assert!(transitions
        .iter()
        .all(|(fence, target, _)| *fence != 1 || *target == DriverRunTarget::Running));
    assert!(transitions
        .iter()
        .any(|(fence, target, _)| *fence == 2 && *target == DriverRunTarget::Running));
    assert_eq!(
        transitions
            .iter()
            .filter(|(fence, target, _)| { *fence == 2 && *target == DriverRunTarget::Succeeded })
            .count(),
        1
    );
    assert_eq!(authorizations.len(), 2);
    assert_ne!(authorizations[0].1, authorizations[1].1);

    let outputs = decrypt_outputs(control.as_ref(), &result_artifact);
    let store = super::object_transfer::RemoteObjectStore::new().unwrap();
    for location in &outputs.objects {
        let artifact = {
            let state = control.state.lock().unwrap();
            state.artifacts.get(&location.artifact_id).unwrap().clone()
        };
        let bytes = open_object(
            &fixture.run_key,
            &artifact.ciphertext,
            RUN_ID,
            EncryptionPurpose::Result,
        )
        .unwrap();
        store
            .register(ProgramExecutionObject {
                reference: location.reference.clone(),
                bytes,
            })
            .unwrap();
    }
    if let StudyExpectation::Meshing(access) = &fixture.expectation {
        let runmat_execution_artifact::ProgramExecutionResponse::ExternalizedSuccess {
            outputs: result_values,
            result_objects,
        } = outputs.response
        else {
            panic!("replaced meshing root did not return an externalized result")
        };
        let [runmat_execution::value::ValuePayload::Object(root)] = result_values.as_slice() else {
            panic!("replaced meshing root returned an invalid result root")
        };
        assert!(result_objects.contains(root));
        let terminal = runmat_meshing_execution::import_exact_meshing_result(
            &store,
            root.as_ref().clone(),
            result_objects,
            access.clone(),
            ObjectInventoryLimits::default(),
        )
        .unwrap();
        assert!(!terminal.artifact.topology.volume_elements.is_empty());
        return None;
    }
    let imported = match &fixture.expectation {
        StudyExpectation::Exact(expectation) => import_exact_structural_study_result(
            &store,
            outputs.response.clone(),
            expectation,
            ObjectInventoryLimits::default(),
        )
        .unwrap(),
        StudyExpectation::Typed(expectation) => {
            let ExactStudyFamily::ThermalPrestress(case) = family else {
                panic!("typed fixture must identify its authored thermal branch");
            };
            thermal::import_cold(case, &store, outputs.response.clone(), expectation)
        }
        StudyExpectation::Meshing(_) => unreachable!("handled above"),
    };
    assert_replacement_result(family, imported.analysis);
    match &fixture.expectation {
        StudyExpectation::Typed(expectation) => Some(thermal::ColdReplay {
            store,
            response: outputs.response,
            expectation: expectation.as_ref().clone(),
            thermal: imported.thermal.expect("typed fixture retains heat result"),
        }),
        StudyExpectation::Exact(_) => None,
        StudyExpectation::Meshing(_) => unreachable!("handled above"),
    }
}

fn assert_replacement_result(
    family: ExactStudyFamily,
    analysis: ExactStructuralStudyAnalysisResult,
) {
    match (family, analysis) {
        (ExactStudyFamily::ThermalPrestress(case), result) => thermal::assert_branch(case, &result),
        (
            ExactStudyFamily::StructuralTransient,
            ExactStructuralStudyAnalysisResult::StructuralTransient {
                result,
                numerical_receipt,
                ..
            },
        ) => {
            assert_eq!(result.samples().len(), 3);
            assert_eq!(numerical_receipt.blocks.len(), 2);
            assert_eq!(result.samples().last().unwrap().schedule_ordinal, 2);
            numerical_receipt.validate_for_result(&result).unwrap();
        }
        (
            ExactStudyFamily::ModalResponseSpectrum,
            ExactStructuralStudyAnalysisResult::StructuralModalResponseSpectrum(result),
        ) => result
            .response_numerical_receipt
            .validate_against(
                result.response_numerical_receipt.source_result(),
                &result.response_result,
            )
            .unwrap(),
        (
            ExactStudyFamily::ModalRandomResponse,
            ExactStructuralStudyAnalysisResult::StructuralRandomResponse(result),
        ) => result
            .response_numerical_receipt
            .validate_against(
                result.response_numerical_receipt.source_result(),
                &result.response_result,
            )
            .unwrap(),
        (expected, actual) => panic!(
            "replacement returned the wrong structural study family: expected {expected:?}, got {actual:?}"
        ),
    }
}

#[path = "driver_replacement_tests/fixture.rs"]
mod fixture;
mod observer;
#[path = "driver_replacement_tests/relay.rs"]
mod relay;
#[path = "driver_replacement_tests/thermal.rs"]
mod thermal;

use fixture::*;
use relay::*;

fn decrypt_outputs(control: &DurableControl, artifact_id: &str) -> ProgramExecutionOutputs {
    let artifact = control
        .state
        .lock()
        .unwrap()
        .artifacts
        .get(artifact_id)
        .unwrap()
        .clone();
    let plaintext = open_object(
        &control.run_key,
        &artifact.ciphertext,
        RUN_ID,
        EncryptionPurpose::Result,
    )
    .unwrap();
    serde_json::from_slice(&plaintext).unwrap()
}

fn artifact_download(id: &str, artifact: &StoredArtifact) -> DriverArtifactDownload {
    DriverArtifactDownload {
        artifact_id: id.into(),
        kind: artifact.kind.clone(),
        ciphertext_digest: ciphertext_digest(&artifact.ciphertext),
        ciphertext_size_bytes: artifact.ciphertext.len() as u64,
        media_type: "application/vnd.runmat.execution+ciphertext".into(),
        method: "GET".into(),
        url: format!("memory://{id}"),
        headers: BTreeMap::new(),
    }
}

const fn artifact_kind(kind: DriverArtifactKind) -> &'static str {
    match kind {
        DriverArtifactKind::Result => "result",
        DriverArtifactKind::Checkpoint => "checkpoint",
        DriverArtifactKind::Diagnostic => "diagnostic",
    }
}

fn unavailable(message: &str) -> runmat_execution_transport_native::TransportError {
    runmat_execution_transport_native::TransportError::Unavailable(message.into())
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
