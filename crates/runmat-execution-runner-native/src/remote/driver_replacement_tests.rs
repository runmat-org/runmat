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
    ProgramExecutionObject, ProgramExecutionObjectLocation, ProgramExecutionOutputs,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
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
    state: Mutex<DurableState>,
    recovery_committed: tokio::sync::Notify,
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
        let committed_recovery_checkpoint = artifact.checkpoint.is_some_and(|(_, disposition)| {
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
            .is_some_and(|checkpoint| checkpoint.closure_catalog.is_some())
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
                && committed_recovery_checkpoint
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
        self.state
            .lock()
            .unwrap()
            .worker_authorizations
            .push((authority.fencing_token, allocation_lease_id.into()));
        Ok(())
    }
}

#[test]
fn actual_driver_restarts_from_server_checkpoint_on_a_different_worker() {
    std::thread::Builder::new()
        .name("actual-driver-replacement".into())
        .stack_size(ACTUAL_DRIVER_TEST_STACK_BYTES)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("actual-driver replacement runtime")
                .block_on(actual_driver_replacement_case());
        })
        .expect("actual-driver replacement test thread")
        .join()
        .expect("actual-driver replacement test thread panicked");
}

async fn actual_driver_replacement_case() {
    let fixture = exact_study_fixture();
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
            .unwrap();
        (
            result,
            state.transitions.clone(),
            state.worker_authorizations.clone(),
            state.bootstrap_checkpoints.clone(),
        )
    };
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
    let imported = import_exact_structural_study_result(
        &store,
        outputs.response,
        &fixture.expectation,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let ExactStructuralStudyAnalysisResult::StructuralTransient {
        result,
        numerical_receipt,
        ..
    } = imported.analysis
    else {
        panic!("replacement returned the wrong structural study family")
    };
    assert_eq!(result.samples().len(), 3);
    assert_eq!(numerical_receipt.blocks.len(), 2);
    assert_eq!(result.samples().last().unwrap().schedule_ordinal, 2);
    numerical_receipt.validate_for_result(&result).unwrap();
}

#[path = "driver_replacement_tests/fixture.rs"]
mod fixture;
#[path = "driver_replacement_tests/relay.rs"]
mod relay;

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
