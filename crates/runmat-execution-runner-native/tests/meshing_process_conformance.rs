#[path = "meshing_process_conformance/attempt_lifecycle.rs"]
mod attempt_lifecycle;
#[path = "support/fea_admission.rs"]
mod fea_admission_support;
use std::collections::{BTreeMap, BTreeSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_execution::resource::{Capability, ResourceInventory};
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ArtifactResult, ExecutableForm, ExecutionBundleBuilder, LogicalObject,
    ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicy, CanonicalMeshingContract, CurveQualityTargets,
    ElementOrder, GeometryRevisionRef, GeometryTolerancePolicy, MeshingCapabilityRequirement,
    MeshingChunkMediaType, MeshingChunkPolicy, MeshingChunkStream, MeshingFailure,
    MeshingInputKind, MeshingInputRef, MeshingPartitionDescriptor, MeshingPartitionKind,
    MeshingProgress, MeshingQualityTargets, MeshingRequest, MeshingSizeLimits,
    MeshingStageIdentity, MeshingStageKind, MeshingWorkloadRequest, MetricCombinationRule,
    MetricFieldRequest, MetricTensor3, NeverCancelled, StableDigest, SurfaceQualityTargets,
    VolumeQualityTargets, MESHING_IDENTITY_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_WORKLOAD_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    build_task_submission, execute_exact_meshing_dag_async, import_result_publication,
    prepare_exact_geometry_input, prepare_exact_geometry_objects, prepare_faceted_geometry_input,
    prepare_faceted_geometry_objects, ExactMeshingDagRun, MeshingExecutionContext,
    MeshingHostResponse, MeshingHostWorkload, MeshingRunEvidenceContext, MeshingStageCheckpoint,
    MeshingStageInvocation, MeshingStageKernel, MeshingTaskEffectPolicy, NoopMeshingProgress,
    PreparedExactGeometryInput, ValidatedMeshingStageOutput,
};

use runmat_execution_runner_native::{
    execute_meshing_program_request, native_meshing_kernel_dispatcher, run_meshing_worker_stdio,
    run_remote_meshing_worker_quic, MeshingExecutionPolicy, NativeExactMeshingExecutor,
    NativeMeshingHostLimits, NativeProgramSession, QuicRemoteWorkerChannel,
    RemoteExactMeshingExecutor, RemoteMeshingWorkerQuicRequest, RemotePoolDriver,
    RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
    NATIVE_OBJECT_STORE_ROOT_ENV,
};

#[path = "meshing_process_conformance/curve_pipeline.rs"]
mod curve_pipeline;
#[path = "meshing_process_conformance/exact_input.rs"]
mod exact_input;
#[path = "meshing_process_conformance/faceted_input.rs"]
mod faceted_input;
#[path = "meshing_process_conformance/remote_recovery.rs"]
mod remote_recovery;

struct AdmissionKernel;

struct PaidAdmissionKernel;

impl MeshingStageKernel for PaidAdmissionKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        assert_eq!(
            runmat_runtime::analysis::fea_run_admission::FeaRunScope::current_allocation(),
            Some(1000),
            "the original paid scope must enter the actual native kernel thread"
        );
        AdmissionKernel.execute(invocation)
    }
}

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
        charge_returned_records(&invocation, 2 * 700);
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

/// Kernels own the byte capacity of the records they return; the stage
/// rejects records whose capacity was never charged to its allocation budget.
fn charge_returned_records(invocation: &MeshingStageInvocation<'_, '_>, bytes: u64) {
    use runmat_geometry_core::GeometryEvaluationControl as _;
    invocation
        .control
        .geometry_evaluation_control()
        .consume_allocation_bytes(bytes)
        .expect("conformance kernel records fit the stage allocation budget");
}

struct SearchBudgetKernel;

struct ExactInputKernel;

struct FacetedInputKernel;

impl MeshingStageKernel for ExactInputKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        assert_eq!(invocation.inputs.len(), 1);
        assert!(invocation.inputs[0].exact_geometry().is_some());
        self.execute_surface(invocation)
    }
}

impl MeshingStageKernel for FacetedInputKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        assert_eq!(invocation.inputs.len(), 1);
        assert!(invocation.inputs[0].faceted_geometry().is_some());
        ExactInputKernel.execute_surface(invocation)
    }
}

impl ExactInputKernel {
    fn execute_surface(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        let checkpoint = MeshingStageCheckpoint {
            completed_work: 1,
            estimated_work: 1,
            peak_memory_bytes: 2048,
            search_work: 1,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;
        charge_returned_records(&invocation, 2 * 700);
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: stable(92),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: 2,
                records: vec![vec![4; 700], vec![5; 700]],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

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
        _invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        // Keep the worker active until its driver retires it. A fixed sleep
        // can finish before the parent requests cancellation on a busy host.
        loop {
            std::thread::park();
        }
    }
}

#[derive(Default)]
struct AdmissionThenCooperativeSlowKernel {
    calls: AtomicUsize,
    cancellation_observed: AtomicBool,
    execution_active: AtomicBool,
}

struct ProductionAndConformanceKernel {
    production_request: StableDigest,
    production: Arc<dyn MeshingStageKernel>,
    conformance: Arc<AdmissionThenCooperativeSlowKernel>,
}

impl MeshingStageKernel for ProductionAndConformanceKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.resolved_request.canonical_digest().ok() == Some(self.production_request)
        {
            self.production.execute(invocation)
        } else {
            self.conformance.execute(invocation)
        }
    }
}

impl MeshingStageKernel for AdmissionThenCooperativeSlowKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage == MeshingStageKind::CurveMesh {
            return curve_pipeline::CurvePipelineKernel::default().execute(invocation);
        }
        if !invocation.inputs.is_empty() {
            return ExactInputKernel.execute(invocation);
        }
        if self.calls.fetch_add(1, Ordering::AcqRel) == 0 {
            return AdmissionKernel.execute(invocation);
        }
        self.execution_active.store(true, Ordering::Release);
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
                self.execution_active.store(false, Ordering::Release);
                return Err(error);
            }
        }
        self.execution_active.store(false, Ordering::Release);
        AdmissionKernel.execute(invocation)
    }
}

#[path = "support/native_host.rs"]
mod native_host;

// witness: FEA-MESH-EXECUTION-001/worker_loss_and_transfer @
//   shared_infrastructure/canonical_mixed_topology/native_process; coverage: partial;
//   boundary: The whole harness-less meshing process-conformance target (worker child processes,
//   progress, cancellation, remote conformance and remote recovery) exits successfully; it is one
//   witness, so the individual claims are not separated.
fn main() {
    native_host::run(run);
}

fn run() {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    if let Some(mode) = arguments.get(1).and_then(|mode| mode.to_str()) {
        if matches!(
            mode,
            "--exit-after-success-child"
                | "--raw-terminal-child"
                | "--retry-observed-child"
                | "--checkpoint-failure-child"
        ) {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(attempt_lifecycle::fault_child(mode));
            return;
        }
    }
    if arguments
        .iter()
        .any(|argument| argument == "--attempt-lifecycle-only")
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(attempt_lifecycle::run());
        return;
    }
    if let Some(mode) = arguments.get(1) {
        if mode == runmat_process_host::HiddenMode::ExecutionDriver.marker() {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(curve_pipeline::run_durable_driver());
            return;
        }
        let root = || {
            std::env::var_os(NATIVE_OBJECT_STORE_ROOT_ENV)
                .expect("child object-store root from native driver")
        };
        if mode == "--child" {
            run_child(AdmissionKernel, Path::new(&root()));
            return;
        }
        if mode == "--paid-child" {
            run_child(PaidAdmissionKernel, Path::new(&root()));
            return;
        }
        if mode == "--slow-child" {
            run_child(SlowKernel, Path::new(&root()));
            return;
        }
        if mode == "--exact-child" {
            run_child(ExactInputKernel, Path::new(&root()));
            return;
        }
        if mode == "--faceted-child" {
            run_child(FacetedInputKernel, Path::new(&root()));
            return;
        }
        if mode == "--curve-child" {
            run_child(
                curve_pipeline::CurvePipelineKernel::default(),
                Path::new(&root()),
            );
            return;
        }
        if mode == "--exact-dag-child" {
            run_child(native_meshing_kernel_dispatcher(), Path::new(&root()));
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
    use runmat_runtime::analysis::fea_run_admission::{
        FeaRunKind, FeaRunOutcome, FeaRunRequest, FeaRunScope,
    };
    let directory = tempfile::tempdir().unwrap();
    let (host, request) = fixture();
    let session = NativeProgramSession::new(config(directory.path(), "--paid-child")).unwrap();
    seed_host_invocation(&mut session.object_store(), &host, &request);
    let task_submission = submission(&session, &host, &request);
    let admission = fea_admission_support::context();
    let scope = admission
        .run_sync(|| {
            FeaRunScope::admit(
                FeaRunRequest::local(
                    FeaRunKind::MeshAndSolve,
                    1000,
                    Digest::sha256(b"paid-meshing-conformance"),
                )
                .unwrap(),
            )
        })
        .unwrap();
    let task = scope
        .run_sync(|| session.submit(request.clone(), task_submission))
        .unwrap();
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
    scope.settle(FeaRunOutcome::Completed).unwrap();
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
    let (outputs, result_objects) = success.into_values().expect("ordinary task result");
    assert!(serde_json::to_vec(&outputs).unwrap().len() < 4096);
    assert!(result_objects.len() >= 3);
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        panic!("native meshing child returned a non-object root")
    };
    let store = session.object_store();
    let imported = import_result_publication(
        &store,
        root,
        host.artifact_access.clone(),
        limits().inventory,
    )
    .unwrap();
    assert!(imported
        .result_objects()
        .iter()
        .all(|reference| result_objects.contains(reference)));
    runmat_meshing_execution::import_stage_evidence_observation(
        &store,
        &host,
        &result_objects,
        limits().inventory,
    )
    .unwrap();

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
    match budget_failure {
        ProgramExecutionResponse::Failure { message } => assert_eq!(
            MeshingHostResponse::failure_from_program_message(&message)
                .expect("canonical meshing budget failure")
                .category,
            runmat_meshing_core::MeshingFailureCategory::SearchWorkBudgetExceeded,
        ),
        response => panic!("search-budget kernel unexpectedly succeeded: {response:?}"),
    }

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

    exact_input::native_conformance().await;
    exact_input::native_dag_conformance();
    faceted_input::native_conformance().await;
    curve_pipeline::native_conformance().await;
    curve_pipeline::durable_conformance().await;

    let cancellation_directory = tempfile::tempdir().unwrap();
    let (cancel_host, cancel_request) = fixture();
    let cancel_session =
        NativeProgramSession::new(config(cancellation_directory.path(), "--slow-child")).unwrap();
    seed_host_invocation(
        &mut cancel_session.object_store(),
        &cancel_host,
        &cancel_request,
    );
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
    assert!(
        matches!(
            &cancelled,
            Err(runmat_execution_runner_native::NativeProgramFailure::Cancelled)
        ),
        "cancelled result: {cancelled:?}"
    );
    drop(cancel_task);
    drop(cancel_session);
    assert!(!cancel_store_root.exists());

    if std::env::args().any(|argument| argument == "--cancellation-only") {
        return;
    }

    remote_conformance().await;
    remote_recovery::run().await;
}

async fn remote_conformance() {
    let (host, request, exact, curve, bundle_bytes) = remote_fixture("remote-meshing-run");
    let full_dag = full_dag_fixture("remote-meshing-run");

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
        host: worker_host_inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([17; 32]).unwrap();
    let kernel = Arc::new(AdmissionThenCooperativeSlowKernel::default());
    let server_kernel = Arc::new(ProductionAndConformanceKernel {
        production_request: full_dag.1.canonical_digest().unwrap(),
        production: Arc::new(native_meshing_kernel_dispatcher()),
        conformance: Arc::clone(&kernel),
    });
    let server = run_remote_meshing_worker_quic(RemoteMeshingWorkerQuicRequest {
        worker: RemoteWorkerQuicRequest {
            listener,
            run_identity: "remote-meshing-run".into(),
            worker: worker.clone(),
            driver_fence: 31,
            session_id: [19; 16],
            run_key: run_key.clone(),
            limits: FrameLimits::default(),
        },
        kernel: server_kernel,
        meshing_limits: limits(),
    });
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
        register_host_invocation(&pool, &host, &request);
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
        let (success_outputs, success_result_objects) =
            success.values().expect("ordinary task result");
        let [ValuePayload::Object(remote_root)] = success_outputs else {
            panic!("remote meshing returned a non-object root")
        };
        let mut remote_store = TestStore::default();
        for reference in success_result_objects {
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
        assert!(imported
            .result_objects()
            .iter()
            .all(|reference| success_result_objects.contains(reference)));
        runmat_meshing_execution::import_stage_evidence_observation(
            &remote_store,
            &host,
            success_result_objects,
            limits().inventory,
        )
        .unwrap();

        for (object, reference) in exact
            .input
            .geometry_objects()
            .objects
            .iter()
            .zip(exact.input.input_objects())
        {
            pool.register_execution_object(reference.clone(), object.bytes.clone())
                .unwrap();
        }
        register_host_invocation(&pool, &exact.host, &exact.request);
        let exact_completion = pool
            .submit(
                submission_for(scope_id, pool_id, &exact.host, &exact.request),
                exact.request.clone(),
            )
            .unwrap();
        let exact_success = tokio::time::timeout(Duration::from_secs(10), exact_completion.wait())
            .await
            .expect("remote exact-input meshing timeout")
            .unwrap();
        let (exact_outputs, exact_result_objects) =
            exact_success.values().expect("ordinary task result");
        let [ValuePayload::Object(exact_root)] = exact_outputs else {
            panic!("remote exact-input stage returned a non-object root")
        };
        let mut exact_result_store = TestStore::default();
        for reference in exact_result_objects {
            exact_result_store.0.insert(
                reference.logical_digest,
                pool.execution_object(reference)
                    .unwrap()
                    .expect("verified remote exact-input result object")
                    .to_vec(),
            );
        }
        import_result_publication(
            &exact_result_store,
            exact_root,
            exact.host.artifact_access.clone(),
            limits().inventory,
        )
        .unwrap();

        curve_pipeline::remote_conformance(&pool, scope_id, pool_id, &curve).await;

        let mut full_executor = RemoteExactMeshingExecutor::new(
            Arc::clone(&pool),
            full_dag.2.clone(),
            // The remote worker advertises one core; the pool rejects larger requests.
            MeshingExecutionPolicy {
                cpu_millicores: worker_inventory().cpu_millicores,
                // Externalized results cross the QUIC worker boundary. Grant
                // their existing inventory bound, not the local-only default
                // of zero egress; relay traffic remains unrequested.
                maximum_egress_bytes: limits().inventory.max_total_bytes,
                ..MeshingExecutionPolicy::default()
            },
            limits().inventory.max_object_bytes,
            limits().inventory,
        )
        .unwrap();
        let full_result = tokio::time::timeout(
            Duration::from_secs(120),
            execute_exact_meshing_dag_async(
                ExactMeshingDagRun {
                    geometry: &full_dag.0,
                    ceilings: runmat_meshing_core::MeshingExecutionCeilings {
                        maximum_wall_time_ms: 120_000,
                        ..dag_ceilings()
                    },
                    request: full_dag.1.clone(),
                    artifact_access: full_dag.2.clone(),
                    named_regions: Default::default(),
                    capability_cohort: Some("native-cohort-v1".into()),
                    program_revision: request.recipe.program_revision.clone(),
                    preferred_edges_per_partition: 3,
                    preferred_faces_per_partition: 2,
                    inventory_limits: limits().inventory,
                    evidence: MeshingRunEvidenceContext {
                        platform: runmat_meshing_core::PlatformBuildIdentity {
                            capability_cohort: "native-cohort-v1".into(),
                            target_triple: "encrypted-remote-test".into(),
                            build_digest: stable(72),
                            exact_kernel_abi: Some("occt-v1".into()),
                        },
                        sizing: Vec::new(),
                        cache_admission: runmat_meshing_core::CacheAdmissionDecision::Admitted,
                    },
                },
                &mut full_executor,
            ),
        )
        .await
        .expect("full remote exact-meshing DAG timed out")
        .unwrap();
        full_result
            .evidence
            .validate(&full_result.artifact)
            .unwrap();
        let elements = full_result.artifact.topology.volume_elements.len() as u64;
        assert!(elements > 4 && elements <= full_dag.1.limits.maximum_elements);
        assert!(!full_executor.drain_progress().is_empty());

        let mut exact_serial_store = TestStore::default();
        for object in &exact.input.geometry_objects().objects {
            exact_serial_store
                .write_verified(object)
                .expect("seed exact serial input");
        }
        seed_host_invocation(&mut exact_serial_store, &exact.host, &exact.request);
        let exact_serial = execute_meshing_program_request(
            &exact.request,
            &mut exact_serial_store,
            &ExactInputKernel,
            &NeverCancelled,
            &mut NoopMeshingProgress,
            limits(),
        );
        let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = exact_serial else {
            panic!("serial exact-input reference did not externalize its result")
        };
        assert_eq!(outputs, exact_outputs);

        let mut serial_store = TestStore::default();
        seed_host_invocation(&mut serial_store, &host, &request);
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
        assert_eq!(outputs, success_outputs);

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
        register_host_invocation(&cancel_pool, &host, &request);
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
        assert!(!kernel.execution_active.load(Ordering::Acquire));
        tokio::time::timeout(Duration::from_secs(5), channel.drain())
            .await
            .expect("remote meshing drain timeout")
            .unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
    assert!(kernel.cancellation_observed.load(Ordering::Acquire));
}

fn full_dag_fixture(
    authorization_scope: &str,
) -> (PreparedExactGeometryInput, MeshingRequest, ArtifactAccess) {
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let (document, topology, evaluators) = runmat_geometry_fixtures::exact_tetrahedron();
    let objects = prepare_exact_geometry_objects(
        document.clone(),
        topology,
        evaluators,
        None,
        None,
        limits().inventory,
    )
    .unwrap();
    let geometry =
        prepare_exact_geometry_input(objects, access.clone(), limits().inventory).unwrap();
    let mut request = request();
    request.tolerance = document.tolerance;
    request.metric.global_metric = MetricTensor3::isotropic_length_m(10.0).unwrap();
    request.quality.curve.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.curve.maximum_tangent_change_degrees = 180.0;
    request.quality.curve.minimum_metric_edge_length = 0.01;
    request.quality.curve.maximum_metric_edge_length = 10.0;
    request.quality.surface.minimum_metric_angle_degrees = 0.1;
    request.quality.surface.maximum_physical_aspect_ratio = 1_000.0;
    request.quality.surface.maximum_chordal_deviation_m = runmat_types::physical::Length::new(0.1);
    request.quality.surface.maximum_normal_deviation_degrees = 180.0;
    request.quality.volume.maximum_metric_edge_length = 2.0;
    request.quality.volume.maximum_radius_edge_ratio = 10.0;
    request.quality.volume.minimum_scaled_jacobian = 0.01;
    request.limits.maximum_nodes = 10_000;
    request.limits.maximum_elements = 10_000;
    (geometry, request, access)
}

fn remote_fixture(
    authorization_scope: &str,
) -> (
    MeshingHostWorkload,
    runmat_execution_artifact::ProgramExecutionRequest,
    exact_input::ExactFixture,
    curve_pipeline::CurveFixture,
    Arc<Vec<u8>>,
) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    let project_document = "[package]\nname = \"remote-meshing\"\n[sources]\nroots = [\"src\"]\n[runtime.language]\ncompat = \"matlab\"\n";
    std::fs::write(project_root.path().join("runmat.toml"), project_document).unwrap();
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
    let exact = exact_input::fixture(revision.clone(), authorization_scope);
    let curve = curve_pipeline::fixture(revision.clone(), authorization_scope);
    let mut materializations = vec![
        (
            request.recipe.clone(),
            request.artifact.executable_bytes.clone(),
        ),
        (
            exact.request.recipe.clone(),
            exact.request.artifact.executable_bytes.clone(),
        ),
        (
            curve.partition_request.recipe.clone(),
            curve.partition_request.artifact.executable_bytes.clone(),
        ),
        (
            curve.join_request.recipe.clone(),
            curve.join_request.artifact.executable_bytes.clone(),
        ),
    ];
    materializations.sort_by_key(|(recipe, _)| recipe.id().unwrap());
    materializations.dedup_by_key(|(recipe, _)| recipe.id().unwrap());
    assert_eq!(materializations.len(), 1);
    let mut builder = ExecutionBundleBuilder::native(&project, revision)
        .unwrap()
        .with_project_runtime_policy_object(
            runmat_core::cluster_settings::capture_project_policy(
                &runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
                    project_document,
                    runmat_config::document::RunmatConfigFormat::Toml,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_compiled_package_closure();
    for (recipe, executable_bytes) in materializations {
        builder = builder.with_materialized_program(
            recipe,
            ExecutableForm::MeshingWorkload,
            executable_bytes,
        );
    }
    let bundle = builder.build().unwrap();
    let mut encoded_bundle = Vec::new();
    write_bundle(&bundle, &mut encoded_bundle, ArchiveLimits::default()).unwrap();
    (host, request, exact, curve, Arc::new(encoded_bundle))
}

#[derive(Default)]
struct TestStore(BTreeMap<Digest, Vec<u8>>);

impl CacheImport for TestStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        match self.0.get(&digest) {
            Some(bytes) if bytes.len() as u64 > maximum_bytes => {
                Err(runmat_execution_artifact::ArtifactError::Limit(
                    "object exceeds its read bound".into(),
                ))
            }
            bytes => Ok(bytes.cloned()),
        }
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

fn seed_host_invocation(
    store: &mut impl CacheExport,
    host: &MeshingHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) {
    let roots = MeshingHostWorkload::semantic_input_roots(request).unwrap();
    let invocation = host
        .prepare_program_request(request.recipe.program_revision.clone(), &roots)
        .unwrap();
    assert_eq!(invocation.request(), request);
    invocation.write_to(store).unwrap();
}

fn register_host_invocation(
    pool: &RemotePoolDriver,
    host: &MeshingHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) {
    let roots = MeshingHostWorkload::semantic_input_roots(request).unwrap();
    let invocation = host
        .prepare_program_request(request.recipe.program_revision.clone(), &roots)
        .unwrap();
    assert_eq!(invocation.request(), request);
    let root = MeshingHostWorkload::invocation_root(request).unwrap();
    assert_eq!(
        root.logical_digest,
        invocation.descriptor().descriptor.digest
    );
    pool.register_execution_object(root.clone(), invocation.descriptor().bytes.clone())
        .unwrap();
}

fn worker_inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: 64_000_000,
        scratch_bytes: 4_000_000,
        accelerators: Vec::new(),
        capabilities: worker_capabilities(),
    }
}

fn worker_host_inventory() -> runmat_execution::host::ExecutionHostInventory {
    runmat_core::RunMatSession::with_options(false, false)
        .unwrap()
        .execution_host_inventory(runmat_execution::security::ExecutionTrustTier::CustomerTrusted)
        .unwrap()
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
        host,
        request,
        BTreeSet::new(),
        &MeshingExecutionContext {
            scope_id,
            pool_id,
            permitted_trust_tiers: BTreeSet::from([
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            ]),
            artifact_access: host.artifact_access.clone(),
            cpu_millicores: 1000,
            // Remote completion transfers its content-addressed artifacts;
            // admit that transfer within the same stage artifact ceiling.
            maximum_egress_bytes: host.ceilings.maximum_artifact_bytes,
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
        Capability::Custom("runmat.meshing.algorithm:curve/v2".into()),
        Capability::Custom("runmat.meshing.algorithm:surface/v2".into()),
        Capability::Custom("runmat.meshing.algorithm:plc/v2".into()),
        Capability::Custom("runmat.meshing.algorithm:tetrahedron/v2".into()),
        Capability::Custom("runmat.meshing.algorithm:optimization/v2".into()),
        Capability::Custom("runmat.meshing.algorithm:validation/v2".into()),
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
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let request = request();
    let identity = MeshingStageIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageKind::GeometryAdmission,
        geometry: GeometryRevisionRef {
            source_unit_evidence:
                runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                    runmat_geometry_core::LengthUnit::Meter,
                    "canonical solver fixture",
                ),
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
    let host = MeshingHostWorkload::new(
        workload,
        identity,
        request,
        fixture_ceilings(),
        access,
        None,
    )
    .unwrap();
    let program = host
        .prepare_program_request(revision, &[])
        .unwrap()
        .into_request();
    (host, program)
}

fn request() -> MeshingRequest {
    MeshingRequest {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: ElementOrder::Tet4,
        dimension: runmat_meshing_core::MeshDimension::Solid,
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
            source_tolerance_m: runmat_types::physical::Length::new(1.0e-8),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-10),
            model_relative_term: 1.0e-9,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-5),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        },
        metric: MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
            cells_across_gap: None,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_tangent_change_degrees: 5.0,
                minimum_metric_edge_length: 0.1,
                maximum_metric_edge_length: 1.5,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.1,
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-5),
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        limits: MeshingSizeLimits {
            maximum_nodes: 100,
            maximum_elements: 100,
            maximum_recursion_depth: 32,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 1000,
            maximum_work_units_between_checks: 1000,
        },
        size_controls: Vec::new(),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"native-meshing-graph"),
        Digest::sha256(b"native-meshing-source"),
        // Native workers advertise this runtime; the program must require the same.
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn limits() -> NativeMeshingHostLimits {
    NativeMeshingHostLimits {
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 1_000_000,
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

/// Execution ceilings of the small conformance fixture.
fn fixture_ceilings() -> runmat_meshing_core::MeshingExecutionCeilings {
    runmat_meshing_core::MeshingExecutionCeilings {
        maximum_memory_bytes: 4_000_000,
        maximum_scratch_bytes: 4_000_000,
        maximum_wall_time_ms: 10_000,
        maximum_artifact_bytes: 1_000_000,
        maximum_search_work: 100,
        maximum_iterations: 100,
    }
}

/// Execution ceilings for the complete exact-meshing DAG fixtures.
fn dag_ceilings() -> runmat_meshing_core::MeshingExecutionCeilings {
    runmat_meshing_core::MeshingExecutionCeilings {
        maximum_memory_bytes: 64_000_000,
        maximum_search_work: 1_000_000,
        maximum_iterations: 1_000_000,
        ..fixture_ceilings()
    }
}
