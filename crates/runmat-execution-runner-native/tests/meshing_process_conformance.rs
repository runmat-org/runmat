use std::collections::{BTreeMap, BTreeSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactResult, ExecutableForm, ExecutionBundleBuilder, LogicalObject, ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CanonicalMeshingContract, ElementOrder,
    GeometryRevisionRef, GeometryTolerancePolicy, MeshingCapabilityRequirement,
    MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStream, MeshingFailure,
    MeshingPartitionDescriptor, MeshingPartitionKind, MeshingProgress, MeshingQualityTargets,
    MeshingRequest, MeshingResourceBudget, MeshingStageIdentity, MeshingStageKind,
    MeshingWorkloadRequest, MetricCombinationRule, MetricFieldRequest, MetricTensor3,
    NeverCancelled, StableDigest, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_IDENTITY_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_WORKLOAD_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    build_task_submission, import_result_publication, MeshingArtifactAccess,
    MeshingExecutionContext, MeshingHostWorkload, MeshingStageCheckpoint, MeshingStageInvocation,
    MeshingStageKernel, MeshingTaskEffectPolicy, NoopMeshingProgress, ValidatedMeshingStageOutput,
};

use runmat_execution_runner_native::{
    execute_meshing_program_request, run_meshing_worker_stdio, run_remote_meshing_worker_quic,
    NativeMeshingHostLimits, NativeProgramSession, QuicRemoteWorkerChannel, RemotePoolDriver,
    RemoteWorkerChannel, RemoteWorkerChannelConfig, NATIVE_OBJECT_STORE_ROOT_ENV,
};

#[path = "meshing_process_conformance/remote_recovery.rs"]
mod remote_recovery;

struct AdmissionKernel;

impl MeshingStageKernel for AdmissionKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        assert!(invocation.inputs.is_empty());
        let mut entity_counts = BTreeMap::new();
        entity_counts.insert("bodies_admitted".into(), 1);
        let checkpoint = MeshingStageCheckpoint {
            completed_work: 1,
            estimated_work: 1,
            peak_memory_bytes: 2048,
            search_work: 1,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: stable(90),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::ExactGeometry,
                schema_version: 2,
                records: vec![vec![1; 700], vec![2; 700]],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

struct SearchBudgetKernel;

impl MeshingStageKernel for SearchBudgetKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        invocation.control.checkpoint(MeshingStageCheckpoint {
            completed_work: 1,
            estimated_work: 1,
            search_work: 101,
            ..MeshingStageCheckpoint::default()
        })?;
        unreachable!("the stage-local search budget must reject this kernel")
    }
}

struct SlowKernel;

impl MeshingStageKernel for SlowKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        std::thread::sleep(Duration::from_secs(5));
        AdmissionKernel.execute(invocation)
    }
}

#[derive(Default)]
struct AdmissionThenCooperativeSlowKernel {
    calls: AtomicUsize,
    cancellation_observed: AtomicBool,
}

impl MeshingStageKernel for AdmissionThenCooperativeSlowKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
            return AdmissionKernel.execute(invocation);
        }
        for work in 1..=100 {
            std::thread::sleep(Duration::from_millis(10));
            let checkpoint = MeshingStageCheckpoint {
                completed_work: work,
                estimated_work: 100,
                search_work: work,
                ..MeshingStageCheckpoint::default()
            };
            if let Err(error) = invocation.control.checkpoint(checkpoint) {
                self.cancellation_observed.store(true, Ordering::Release);
                return Err(error);
            }
        }
        AdmissionKernel.execute(invocation)
    }
}

fn main() {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    if let Some(mode) = arguments.get(1) {
        let root = std::env::var_os(NATIVE_OBJECT_STORE_ROOT_ENV)
            .expect("child object-store root from native driver");
        if mode == "--child" {
            run_child(AdmissionKernel, Path::new(&root));
            return;
        }
        if mode == "--slow-child" {
            run_child(SlowKernel, Path::new(&root));
            return;
        }
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(parent());
}

fn run_child(kernel: impl MeshingStageKernel + 'static, root: &Path) {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(run_meshing_worker_stdio(
            std::sync::Arc::new(kernel),
            root,
            limits(),
        ))
        .unwrap();
}

async fn parent() {
    let directory = tempfile::tempdir().unwrap();
    let (host, request) = fixture();
    let session = NativeProgramSession::new(config(directory.path(), "--child")).unwrap();
    let task_submission = submission(&session, &host, &request);
    let task = session.submit(request.clone(), task_submission).unwrap();
    let success = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let progress = task.drain_progress();
    assert!(progress.len() >= 3);
    let decoded_progress = progress
        .iter()
        .map(|progress| {
            assert_eq!(
                progress.media_type,
                "application/vnd.runmat.meshing-progress+cbor"
            );
            MeshingProgress::canonical_decode(&progress.payload).unwrap()
        })
        .collect::<Vec<_>>();
    assert!(decoded_progress
        .windows(2)
        .all(|pair| pair[0].sequence < pair[1].sequence));
    assert_eq!(decoded_progress.last().unwrap().completed_work, 1);
    let outputs = success.outputs;
    let result_objects = success.result_objects;
    assert!(serde_json::to_vec(&outputs).unwrap().len() < 4096);
    assert!(result_objects.len() >= 3);
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        panic!("native meshing child returned a non-object root")
    };
    let store = session.object_store();
    let imported =
        import_result_publication(&store, root, host.artifact_access, limits().inventory).unwrap();
    assert_eq!(imported.result_objects(), result_objects);

    let mut progress = NoopMeshingProgress;
    let mut local_store = store.clone();
    let budget_failure = execute_meshing_program_request(
        &request,
        &mut local_store,
        &SearchBudgetKernel,
        &NeverCancelled,
        &mut progress,
        limits(),
    );
    assert!(matches!(
        budget_failure,
        ProgramExecutionResponse::Failure { message }
            if message.contains("SearchWorkBudgetExceeded")
    ));

    let mut malformed = request;
    malformed.artifact.executable_bytes.push(0);
    let rejected = execute_meshing_program_request(
        &malformed,
        &mut local_store,
        &AdmissionKernel,
        &NeverCancelled,
        &mut progress,
        limits(),
    );
    assert!(matches!(rejected, ProgramExecutionResponse::Failure { .. }));

    let cancellation_directory = tempfile::tempdir().unwrap();
    let (cancel_host, cancel_request) = fixture();
    let cancel_session =
        NativeProgramSession::new(config(cancellation_directory.path(), "--slow-child")).unwrap();
    let cancel_store_root = cancel_session.object_store().root().to_path_buf();
    let cancel_task = cancel_session
        .submit(
            cancel_request.clone(),
            submission(&cancel_session, &cancel_host, &cancel_request),
        )
        .unwrap();
    let live_progress = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let progress = cancel_task.drain_progress();
            if !progress.is_empty() {
                break progress;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(live_progress
        .iter()
        .all(|progress| MeshingProgress::canonical_decode(&progress.payload).is_ok()));
    assert!(cancel_task.try_result().is_none());
    cancel_session.cancel(runmat_execution::CancellationReason::User);
    let cancelled = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(result) = cancel_task.try_result() {
                break result;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert!(matches!(cancelled, Err(message) if message.contains("cancelled")));
    drop(cancel_task);
    drop(cancel_session);
    assert!(!cancel_store_root.exists());

    remote_conformance().await;
    remote_recovery::run().await;
}

async fn remote_conformance() {
    let (host, request, bundle_bytes) = remote_fixture("remote-meshing-run");

    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-meshing-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"remote-meshing-pool"]);
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-meshing-worker"]),
        pool_id,
        resources: worker_inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([17; 32]).unwrap();
    let kernel = Arc::new(AdmissionThenCooperativeSlowKernel::default());
    let server_kernel = Arc::clone(&kernel);
    let server = run_remote_meshing_worker_quic(
        listener,
        "remote-meshing-run",
        worker.clone(),
        31,
        [19; 16],
        run_key.clone(),
        FrameLimits::default(),
        server_kernel,
        limits(),
    );
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-meshing-run".into(),
                node_identity: "remote-meshing-node".into(),
                worker,
                driver_fence: 31,
                session_id: [19; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority,
                server_name: "runmat.execution".into(),
                certificate_der,
            },
        )
        .await
        .unwrap();
        let pool = RemotePoolDriver::new_with_value_scope(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: worker_inventory(),
            },
            31,
            bundle_bytes.as_ref().clone(),
            "remote-meshing-run",
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        let completion = pool
            .submit(
                submission_for(scope_id, pool_id, &host, &request),
                request.clone(),
            )
            .unwrap();
        let remote_progress = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let progress = completion.drain_progress();
                if !progress.is_empty() {
                    break progress;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("remote meshing progress timeout");
        let decoded_remote_progress = remote_progress
            .iter()
            .map(|progress| MeshingProgress::canonical_decode(&progress.payload).unwrap())
            .collect::<Vec<_>>();
        assert!(decoded_remote_progress
            .windows(2)
            .all(|pair| pair[0].sequence < pair[1].sequence));
        let success = tokio::time::timeout(Duration::from_secs(10), completion.wait())
            .await
            .expect("remote meshing completion timeout")
            .unwrap();
        let [ValuePayload::Object(remote_root)] = success.outputs.as_slice() else {
            panic!("remote meshing returned a non-object root")
        };
        let mut remote_store = TestStore::default();
        for reference in &success.result_objects {
            remote_store.0.insert(
                reference.logical_digest,
                pool.execution_object(reference)
                    .unwrap()
                    .expect("verified remote result object")
                    .to_vec(),
            );
        }
        let imported = import_result_publication(
            &remote_store,
            remote_root,
            host.artifact_access.clone(),
            limits().inventory,
        )
        .unwrap();
        assert_eq!(imported.result_objects(), success.result_objects);

        let mut serial_store = TestStore::default();
        let serial = execute_meshing_program_request(
            &request,
            &mut serial_store,
            &AdmissionKernel,
            &NeverCancelled,
            &mut NoopMeshingProgress,
            limits(),
        );
        let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = serial else {
            panic!("serial meshing reference did not externalize its result")
        };
        assert_eq!(outputs, success.outputs);

        let cancel_scope =
            runmat_execution::ExecutionScopeId::derive(&[b"remote-meshing-cancel-scope"]);
        let cancel_pool = RemotePoolDriver::new_with_value_scope(
            cancel_scope,
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: worker_inventory(),
            },
            31,
            bundle_bytes.as_ref().clone(),
            "remote-meshing-run",
        )
        .unwrap();
        cancel_pool.add_worker(channel.clone()).await.unwrap();
        let cancelled = cancel_pool
            .submit(
                submission_for(cancel_scope, pool_id, &host, &request),
                request,
            )
            .unwrap();
        let live_remote_progress = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let progress = cancelled.drain_progress();
                if !progress.is_empty() {
                    break progress;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("live remote meshing progress timeout");
        assert!(live_remote_progress
            .iter()
            .all(|progress| MeshingProgress::canonical_decode(&progress.payload).is_ok()));
        cancel_pool
            .cancel(runmat_execution::CancellationReason::User)
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(2), cancelled.wait())
                .await
                .expect("remote cancellation completion timeout")
                .is_err()
        );
        tokio::time::timeout(Duration::from_secs(5), channel.drain())
            .await
            .expect("remote meshing drain timeout")
            .unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
    assert!(kernel.cancellation_observed.load(Ordering::Acquire));
}

fn remote_fixture(
    authorization_scope: &str,
) -> (
    MeshingHostWorkload,
    runmat_execution_artifact::ProgramExecutionRequest,
    Arc<Vec<u8>>,
) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-meshing\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let (host, request) = fixture_for(revision.clone(), authorization_scope);
    let bundle = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            request.recipe.clone(),
            ExecutableForm::MeshingWorkload,
            request.artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let mut encoded_bundle = Vec::new();
    write_bundle(&bundle, &mut encoded_bundle, ArchiveLimits::default()).unwrap();
    (host, request, Arc::new(encoded_bundle))
}

#[derive(Default)]
struct TestStore(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for TestStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }
}

impl CacheExport for TestStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        self.0
            .insert(object.descriptor.digest, object.bytes.clone());
        Ok(())
    }
}

fn worker_inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: 4_000_000,
        scratch_bytes: 4_000_000,
        accelerators: Vec::new(),
        capabilities: worker_capabilities(),
    }
}

fn config(root: &Path, worker_mode: &str) -> runmat_execution_runner_native::NativeExecutionConfig {
    let mut config =
        runmat_execution_runner_native::NativeExecutionConfig::for_current_executable().unwrap();
    config.executable = std::env::current_exe().unwrap();
    config.worker_arguments = vec![worker_mode.into()];
    config.max_workers = 1;
    config.store_root = root.join("native-session");
    config.worker_capabilities = worker_capabilities();
    config
}

fn submission(
    session: &NativeProgramSession,
    host: &MeshingHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) -> runmat_execution_runner::TaskSubmission {
    submission_for(session.scope_id(), session.pool_id(), host, request)
}

fn submission_for(
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    host: &MeshingHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) -> runmat_execution_runner::TaskSubmission {
    build_task_submission(
        &host.workload,
        &host.stage_identity,
        &host.resolved_request,
        &[],
        BTreeSet::new(),
        &MeshingExecutionContext {
            scope_id,
            pool_id,
            program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
            artifact_access: host.artifact_access.clone(),
            cpu_millicores: 1000,
            maximum_egress_bytes: 0,
            maximum_relay_bytes: 0,
            deadline_unix_millis: None,
            priority: 0,
        },
        MeshingTaskEffectPolicy::ContentAddressedPure {
            maximum_attempts: 2,
            replay_proof_digest: stable(91),
        },
    )
    .unwrap()
}

fn worker_capabilities() -> BTreeSet<Capability> {
    BTreeSet::from([
        Capability::ProcessIsolation,
        Capability::Custom("runmat.meshing.host:host-v2".into()),
        Capability::Custom("runmat.meshing.exact-cad:occt-v1".into()),
        Capability::Custom("runmat.meshing.algorithm:geometry/v2".into()),
        Capability::Custom("runmat.meshing.element-order:tet4".into()),
        Capability::Custom("runmat.meshing.cohort:native-cohort-v1".into()),
    ])
}

fn fixture() -> (
    MeshingHostWorkload,
    runmat_execution_artifact::ProgramExecutionRequest,
) {
    fixture_for(revision(), "native-meshing-run")
}

fn fixture_for(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> (
    MeshingHostWorkload,
    runmat_execution_artifact::ProgramExecutionRequest,
) {
    let access = MeshingArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let request = request();
    let identity = MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::GeometryAdmission,
        geometry: GeometryRevisionRef {
            source_digest: stable(1),
            geometry_revision: 2,
            persistent_mapping_version: 1,
        },
        resolved_request_digest: request.canonical_digest().unwrap(),
        tolerance_policy_digest: request.tolerance.canonical_digest().unwrap(),
        metric_policy_digest: request.metric.canonical_digest().unwrap(),
        algorithm_set_digest: request.algorithms.canonical_digest().unwrap(),
        deterministic_seed: request.deterministic_seed,
        prerequisites: Vec::new(),
        capability_cohort: Some("native-cohort-v1".into()),
    };
    let workload = MeshingWorkloadRequest {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageKind::GeometryAdmission,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition: MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        inputs: Vec::new(),
        required_capabilities: vec![
            MeshingCapabilityRequirement::HostWorkload {
                abi: "host-v2".into(),
            },
            MeshingCapabilityRequirement::ExactCadKernel {
                abi: "occt-v1".into(),
            },
            MeshingCapabilityRequirement::MeshingAlgorithm {
                version: "geometry/v2".into(),
            },
            MeshingCapabilityRequirement::ElementOrder {
                order: ElementOrder::Tet4,
            },
            MeshingCapabilityRequirement::DeterministicPlatformCohort {
                cohort: "native-cohort-v1".into(),
            },
        ],
    };
    let host = MeshingHostWorkload::new(workload, identity, request, access, None).unwrap();
    let program = host.program_request(revision, &[]).unwrap();
    (host, program)
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet4,
        deterministic_seed: 7,
        algorithms: AlgorithmVersionSet {
            geometry: "geometry/v2".into(),
            curve: "curve/v2".into(),
            surface: "surface/v2".into(),
            plc: "plc/v2".into(),
            tetrahedron: "tetrahedron/v2".into(),
            optimization: "optimization/v2".into(),
            validation: "validation/v2".into(),
        },
        tolerance: GeometryTolerancePolicy {
            source_tolerance_m: 1.0e-8,
            absolute_floor_m: 1.0e-10,
            model_relative_term: 1.0e-9,
            requested_deviation_m: 1.0e-5,
            maximum_healing_displacement_m: 1.0e-6,
        },
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        resources: MeshingResourceBudget {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_memory_bytes: 4_000_000,
            maximum_scratch_bytes: 4_000_000,
            maximum_wall_time_ms: 10_000,
            maximum_artifact_bytes: 1_000_000,
            maximum_search_work: 100,
            maximum_recursion_depth: 32,
            maximum_iterations: 100,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 1000,
            maximum_work_units_between_checks: 1000,
        },
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"native-meshing-graph"),
        Digest::sha256(b"native-meshing-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"native-meshing-runtime"),
            Digest::sha256(b"native-meshing-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn limits() -> NativeMeshingHostLimits {
    NativeMeshingHostLimits {
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 1024,
            maximum_records_per_chunk: 10,
            maximum_total_encoded_bytes: 1_000_000,
        },
        inventory: runmat_execution_artifact::object::ObjectInventoryLimits {
            max_objects: 100,
            max_object_bytes: 1_000_000,
            max_total_bytes: 10_000_000,
        },
        max_message_bytes: 1024 * 1024,
    }
}

fn stable(seed: u8) -> StableDigest {
    StableDigest::from_bytes([seed; 32])
}
