use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{SinkExt as _, StreamExt as _};
use runmat_analysis_execution::StructuralStudyWorkloads;
use runmat_analysis_fea::execution::{StructuralTransientBlockPlan, StructuralTransientWorkload};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::resource::ResourceInventory;
use runmat_execution::security::{
    EndpointIdentityEvidence, EndpointIdentitySigner, EndpointRecipientKey, ExecutionTrustTier,
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
    ProgramExecutionOutputs, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_execution_runner::WorkerSpec;
use runmat_execution_transport_native::control::{
    CheckpointDisposition, DriverArtifactDownload, DriverArtifactKind, DriverAuthority,
    DriverBootstrap, DriverControlPlane, DriverHeartbeat, DriverRunTarget, DriverRunTransition,
    DriverWorkerAllocation, DriverWorkerPool, ResourceRequest, StoreDriverArtifact,
    StoredDriverArtifact,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::identity::EndpointIdentityMaterial;
use runmat_geometry_io::ExactCadImportOptions;
use runmat_meshing_core::{MeshingRequest, MetricTensor3};

use super::checkpoint::RemoteDriverCheckpoint;
use super::config::RemoteDriverConfig;
use super::crypto::{ciphertext_digest, open_object, project_revision_identity, seal_object};
use super::worker_entry::{run_remote_meshing_worker_relay_cached, RemoteWorkerRelayRequest};
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
    driver_interrupted_at_commit: bool,
    transitions: Vec<(u64, DriverRunTarget, Option<String>)>,
    worker_authorizations: Vec<(u64, String)>,
    bootstrap_checkpoints: Vec<(u64, Option<String>)>,
}

struct DurableControl {
    project_revision: String,
    endpoint_fingerprint: String,
    run_key_envelope: Vec<u8>,
    run_key: RunKeyMaterial,
    resources: ResourceRequest,
    workers: BTreeMap<u64, DriverWorkerAllocation>,
    interruption: ReplacementInterruption,
    state: Mutex<DurableState>,
    recovery_committed: tokio::sync::Notify,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplacementInterruption {
    RecoveryClosure,
    ReplayableExecutionStart,
    ThermalCommitted,
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

fn committed_thermal_task(
    control: &DurableControl,
    state: &DurableState,
    checkpoint: &RemoteDriverCheckpoint,
) -> Option<runmat_execution::TaskId> {
    use runmat_analysis_execution::AnalysisHostWorkload;
    use runmat_analysis_fea::execution::AnalysisWorkKind;
    use runmat_execution::value::{ValuePayload, ValueRefKind};

    let (pool, closure) = stored_pool_checkpoint(control, state, checkpoint)?;
    for (task_id, record) in &pool.test_snapshot().tasks {
        if record.committed.is_none() {
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
            .is_ok_and(|host| host.workload.kind == AnalysisWorkKind::SolveThermal)
        {
            return Some(*task_id);
        }
    }
    None
}

#[async_trait]
impl DriverControlPlane for DurableControl {
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
            driver_resources: self.resources.clone(),
            desired_worker_count: 1,
            worker_resources: self.resources.clone(),
        })
    }

    async fn heartbeat(
        &self,
        _authority: &DriverAuthority,
        _ttl_seconds: u64,
    ) -> runmat_execution_transport_native::TransportResult<DriverHeartbeat> {
        Ok(DriverHeartbeat {
            run_state: "running".into(),
            cancellation_requested: false,
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
        let committed_replacement_checkpoint =
            artifact.checkpoint.is_some_and(|(_, disposition)| {
                if disposition != CheckpointDisposition::ResumeSafe {
                    return false;
                }
                open_object(
                    &self.run_key,
                    artifact.ciphertext,
                    RUN_ID,
                    EncryptionPurpose::Checkpoint,
                )
                .ok()
                .and_then(|bytes| RemoteDriverCheckpoint::from_canonical_bytes(&bytes).ok())
                .is_some_and(|checkpoint| match self.interruption {
                    ReplacementInterruption::RecoveryClosure => {
                        checkpoint.closure_catalog.is_some()
                    }
                    ReplacementInterruption::ThermalCommitted => {
                        let state = self.state.lock().unwrap();
                        committed_thermal_task(self, &state, &checkpoint).is_some()
                    }
                    ReplacementInterruption::ReplayableExecutionStart => false,
                })
            });
        let (stored, interrupt_driver) = {
            let mut state = self.state.lock().unwrap();
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
            let interrupt_driver = authority.fencing_token == 1
                && committed_replacement_checkpoint
                && !state.driver_interrupted_at_commit;
            if interrupt_driver {
                state.driver_interrupted_at_commit = true;
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
        if interrupt_driver {
            self.recovery_committed.notify_waiters();
            return std::future::pending().await;
        }
        Ok(stored)
    }

    async fn transition(
        &self,
        authority: &DriverAuthority,
        transition: DriverRunTransition,
    ) -> runmat_execution_transport_native::TransportResult<()> {
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
                && !state.driver_interrupted_at_commit;
            if interrupt_driver {
                state.driver_interrupted_at_commit = true;
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
            ReplacementInterruption::ThermalCommitted,
        );
        assert!(result.is_some());
    }
}

fn run_actual_driver_replacement(family: ExactStudyFamily) -> Option<thermal::ColdReplay> {
    run_actual_driver_replacement_with_interruption(family, family.interruption())
}

fn run_actual_driver_replacement_with_interruption(
    family: ExactStudyFamily,
    interruption: ReplacementInterruption,
) -> Option<thermal::ColdReplay> {
    std::thread::Builder::new()
        .name(format!("actual-driver-replacement-{family:?}"))
        .stack_size(ACTUAL_DRIVER_TEST_STACK_BYTES)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actual-driver replacement runtime")
                .block_on(actual_driver_replacement_case(family, interruption))
        })
        .expect("actual-driver replacement test thread")
        .join()
        .expect("actual-driver replacement test thread panicked")
}

async fn actual_driver_replacement_case(
    family: ExactStudyFamily,
    interruption: ReplacementInterruption,
) -> Option<thermal::ColdReplay> {
    let mut fixture = exact_study_fixture(family);
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
    let first_relay = RelayServer::bind().await;
    let first_authority = driver_authority(1, first_relay.http_base());
    let first_worker = run_worker(
        first_relay.ws_url("worker"),
        control.workers.get(&1).unwrap(),
        &fixture.inventory,
        fixture.run_key.clone(),
        1,
    );
    let first_driver = super::driver::run_remote_driver(
        RemoteDriverConfig {
            authority: first_authority,
            endpoint_identity_file: identity_path.clone(),
        },
        control.clone(),
    );
    interrupt_generation_after_commit(
        first_relay.forward_pair(),
        first_worker,
        first_driver,
        control.recovery_committed.notified(),
    )
    .await;

    let checkpoint_after_loss = {
        let state = control.state.lock().unwrap();
        assert!(state.driver_interrupted_at_commit);
        state.latest_checkpoint.clone().unwrap()
    };
    let thermal_before = if interruption == ReplacementInterruption::ThermalCommitted {
        let state = control.state.lock().unwrap();
        let checkpoint = stored_driver_checkpoint(&control, &state, &checkpoint_after_loss);
        let task_id = committed_thermal_task(&control, &state, &checkpoint)
            .expect("driver loss must follow an authenticated thermal task commit");
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint)
            .expect("committed thermal pool checkpoint");
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
        Some((task_id, pool.test_snapshot().clone(), programs))
    } else {
        None
    };
    let second_relay = RelayServer::bind().await;
    let second_authority = driver_authority(2, second_relay.http_base());
    let second_worker = run_worker(
        second_relay.ws_url("worker"),
        control.workers.get(&2).unwrap(),
        &fixture.inventory,
        fixture.run_key.clone(),
        2,
    );
    let second_driver = super::driver::run_remote_driver(
        RemoteDriverConfig {
            authority: second_authority,
            endpoint_identity_file: identity_path,
        },
        control.clone(),
    );
    let driver_result =
        drive_generation(second_relay.forward_pair(), second_worker, second_driver).await;
    driver_result.unwrap();

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
    if let Some((task_id, before, prefix_programs)) = thermal_before {
        let state = control.state.lock().unwrap();
        let final_id = state
            .latest_checkpoint
            .as_ref()
            .expect("final pool checkpoint");
        let checkpoint = stored_driver_checkpoint(&control, &state, final_id);
        let (pool, _) = stored_pool_checkpoint(&control, &state, &checkpoint)
            .expect("final scheduler authority is published");
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
