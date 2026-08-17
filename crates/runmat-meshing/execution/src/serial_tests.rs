use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::task::{Context, Poll, Waker};

use crate::{
    build_task_submission, prepare_result_publication, prepare_stage_objects,
    MeshingArtifactAccess, MeshingExecutionContext, MeshingHostResponseV2, MeshingHostWorkloadV2,
    MeshingTaskEffectPolicy,
};
use runmat_execution::identity::{ArtifactId, AttemptId, WorkerId};
use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{Digest, ExecutionScopeId, PoolId, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactError, ArtifactResult, LogicalObject};
use runmat_execution_runner::backend::SerialBackend;
use runmat_execution_runner::port::BackendPort;
use runmat_execution_runner::{AttemptReport, AttemptRequest};
use runmat_meshing_core::{
    build_chunked_stage_payload, build_closed_stage_manifest, AlgorithmVersionSet,
    CancellationPolicyV2, CanonicalMeshingContract, GeometryRevisionRef, GeometryTolerancePolicy,
    MeshElementOrderV2, MeshingCancellationSignal, MeshingCapabilityRequirementV2,
    MeshingChunkMediaTypeV2, MeshingChunkPolicyV2, MeshingChunkStreamV2, MeshingFailure,
    MeshingFailureCategory, MeshingManifestDispositionV2, MeshingPartitionDescriptorV2,
    MeshingPartitionKindV2, MeshingQualityTargetsV2, MeshingRequestV2, MeshingResourceBudgetV2,
    MeshingStageIdentityV2, MeshingStageResultKindV2, MeshingStageV2, MeshingWorkloadRequestV2,
    MetricCombinationRule, MetricFieldRequestV2, MetricTensor3, NeverCancelled, StableDigest,
    SurfaceQualityTargetsV2, VolumeQualityTargetsV2, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_REQUEST_SCHEMA_VERSION, MESHING_WORKLOAD_SCHEMA_VERSION,
};

use super::{
    execute_serial_stage, MeshingProgressSink, MeshingSerialExecutionError, MeshingStageCheckpoint,
    MeshingStageControl, MeshingStageInvocation, MeshingStageKernel, ValidatedMeshingStageOutput,
};

#[derive(Default)]
struct MemoryStore {
    objects: HashMap<Digest, Vec<u8>>,
}

impl CacheImport for MemoryStore {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        Ok(self.objects.get(&digest).cloned())
    }
}

impl CacheExport for MemoryStore {
    fn write_verified(&mut self, object: &LogicalObject) -> ArtifactResult<()> {
        object.validate()?;
        if let Some(existing) = self.objects.get(&object.descriptor.digest) {
            if existing != &object.bytes {
                return Err(ArtifactError::Identity(
                    "test store content identity collision".into(),
                ));
            }
        } else {
            self.objects
                .insert(object.descriptor.digest, object.bytes.clone());
        }
        Ok(())
    }
}

#[derive(Default)]
struct Progress(Vec<runmat_meshing_core::MeshingProgressV2>);

impl MeshingProgressSink for Progress {
    fn record(&mut self, progress: &runmat_meshing_core::MeshingProgressV2) {
        self.0.push(progress.clone());
    }
}

struct SurfaceKernel;

impl MeshingStageKernel for SurfaceKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        assert_eq!(invocation.inputs.len(), 1);
        let mut counts = BTreeMap::new();
        counts.insert("faces_completed".into(), 1);
        invocation.control.checkpoint(MeshingStageCheckpoint {
            completed_work: 1,
            estimated_work: 3,
            peak_memory_bytes: 1024,
            search_work: 2,
            entity_counts: counts.clone(),
            ..MeshingStageCheckpoint::default()
        })?;
        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: stable(90),
            streams: vec![MeshingChunkStreamV2 {
                media_type: MeshingChunkMediaTypeV2::SurfacePartitions,
                schema_version: 2,
                records: vec![vec![1; 700], vec![2; 700], vec![3; 700]],
            }],
            final_checkpoint: MeshingStageCheckpoint {
                completed_work: 3,
                estimated_work: 3,
                peak_memory_bytes: 2048,
                search_work: 3,
                entity_counts: counts,
                ..MeshingStageCheckpoint::default()
            },
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
        unreachable!("the hard search-work budget must fail before stage output")
    }
}

#[derive(Default)]
struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn serial_stage_imports_executes_and_externalizes_a_validated_closure() {
    let mut fixture = Fixture::new();
    let mut progress = Progress::default();
    let completed = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut progress,
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    completed
        .workload_result()
        .validate_against(&fixture.host.workload)
        .unwrap();
    let response = MeshingHostResponseV2::completed(&fixture.host, &completed).unwrap();
    let encoded = response.canonical_encode().unwrap();
    let decoded = MeshingHostResponseV2::canonical_decode(&encoded).unwrap();
    decoded.validate_against(&fixture.host).unwrap();
    assert_eq!(decoded.attempt_success(), Some(completed.attempt_success()));
    decoded
        .program_response()
        .validate_against(&fixture.program)
        .unwrap();
    assert_eq!(
        completed
            .publication()
            .stage_objects()
            .manifest
            .chunks
            .len(),
        3
    );
    assert!(completed
        .publication()
        .stage_objects()
        .objects
        .iter()
        .all(|object| fixture
            .store
            .objects
            .contains_key(&object.descriptor.digest)));
    assert_eq!(progress.0.len(), 3);
    assert!(progress
        .0
        .windows(2)
        .all(|pair| pair[1].validate_after(&pair[0]).is_ok()));
}

#[test]
fn serial_stage_content_identity_is_independent_of_legal_chunk_size() {
    let mut fine = Fixture::new();
    let mut coarse = Fixture::new();
    let fine = execute_serial_stage(
        &fine.program,
        &mut fine.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let coarse = execute_serial_stage(
        &coarse.program,
        &mut coarse.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(4096),
        ObjectInventoryLimits::default(),
    )
    .unwrap();

    assert_eq!(
        fine.publication()
            .stage_objects()
            .result_identity
            .logical_content_digest,
        coarse
            .publication()
            .stage_objects()
            .result_identity
            .logical_content_digest
    );
    assert_ne!(
        fine.publication().stage_objects().root.digest,
        coarse.publication().stage_objects().root.digest
    );
}

#[test]
fn ordinary_serial_backend_carries_the_complete_fenced_stage_publication() {
    let mut fixture = Fixture::new();
    let completed = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let expected = completed.attempt_success();
    let root = root(&fixture.program.arguments[0]);
    let scope_id = ExecutionScopeId::derive(&[b"serial-stage-scope"]);
    let pool_id = PoolId::derive(&[b"serial-stage-pool"]);
    let submission = build_task_submission(
        &fixture.host.workload,
        &fixture.host.stage_identity,
        &fixture.host.resolved_request,
        std::slice::from_ref(&root),
        BTreeSet::new(),
        &MeshingExecutionContext {
            scope_id,
            pool_id,
            program_artifact_id: ArtifactId::derive(&[fixture.program.artifact.id.0.bytes()]),
            artifact_access: fixture.host.artifact_access.clone(),
            cpu_millicores: 1000,
            maximum_egress_bytes: 0,
            maximum_relay_bytes: 0,
            deadline_unix_millis: None,
            priority: 0,
        },
        MeshingTaskEffectPolicy::UnknownEffect,
    )
    .unwrap();
    let attempt = AttemptRequest {
        id: AttemptId::derive(&[b"serial-stage-attempt"]),
        task_id: submission.request.id,
        scope_id,
        worker_id: WorkerId::derive(&[b"serial-stage-worker"]),
        ordinal: 1,
        driver_fence: 7,
        task: submission.request,
    };
    let mut backend = SerialBackend::new(move |_: &AttemptRequest| {
        Ok(AttemptReport::Succeeded {
            result: expected.clone(),
        })
    });
    let mut future = backend.launch(attempt);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let Poll::Ready(report) = Future::poll(future.as_mut(), &mut context) else {
        panic!("serial backend unexpectedly suspended")
    };
    let report = report.unwrap();
    assert!(matches!(
        report.report,
        AttemptReport::Succeeded { result } if result == completed.attempt_success()
    ));
}

#[test]
fn serial_stage_reports_typed_cancellation_and_search_budget_failures() {
    let mut cancelled = Fixture::new();
    let error = execute_serial_stage(
        &cancelled.program,
        &mut cancelled.store,
        &SurfaceKernel,
        &Cancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    error
        .workload_result()
        .unwrap()
        .validate_against(&cancelled.host.workload)
        .unwrap();
    let response = MeshingHostResponseV2::failed(&cancelled.host, &error)
        .unwrap()
        .unwrap();
    response.validate_against(&cancelled.host).unwrap();
    assert_eq!(
        stage_failure(&error).category,
        MeshingFailureCategory::Cancelled
    );

    let mut bounded = Fixture::new();
    let error = execute_serial_stage(
        &bounded.program,
        &mut bounded.store,
        &SearchBudgetKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert_eq!(
        stage_failure(&error).category,
        MeshingFailureCategory::SearchWorkBudgetExceeded
    );
}

#[test]
fn host_response_rejects_missing_roots_and_wrong_authority() {
    let mut fixture = Fixture::new();
    let completed = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let response = MeshingHostResponseV2::completed(&fixture.host, &completed).unwrap();
    let MeshingHostResponseV2::Validated {
        schema_version,
        stage_manifest_digest,
        mut root,
        mut result_objects,
    } = response
    else {
        panic!("successful stage returned a failure response")
    };
    result_objects.retain(|object| object != &root);
    let missing = MeshingHostResponseV2::Validated {
        schema_version,
        stage_manifest_digest,
        root: root.clone(),
        result_objects,
    };
    assert!(missing.validate_against(&fixture.host).is_err());

    root.authorization_scope = "another-run".into();
    let wrong_authority = MeshingHostResponseV2::Validated {
        schema_version,
        stage_manifest_digest,
        root,
        result_objects: completed.publication().result_objects().to_vec(),
    };
    assert!(wrong_authority.validate_against(&fixture.host).is_err());
}

#[test]
fn stage_control_enforces_every_algorithm_local_resource_counter() {
    let cases = [
        (
            MeshingFailureCategory::NodeBudgetExceeded,
            MeshingStageCheckpoint {
                node_count: 101,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::ElementBudgetExceeded,
            MeshingStageCheckpoint {
                element_count: 101,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::MemoryBudgetExceeded,
            MeshingStageCheckpoint {
                peak_memory_bytes: 4_000_001,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::ScratchBudgetExceeded,
            MeshingStageCheckpoint {
                peak_scratch_bytes: 4_000_001,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::SearchWorkBudgetExceeded,
            MeshingStageCheckpoint {
                search_work: 101,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::RecursionBudgetExceeded,
            MeshingStageCheckpoint {
                recursion_depth: 33,
                ..checkpoint()
            },
        ),
        (
            MeshingFailureCategory::IterationBudgetExceeded,
            MeshingStageCheckpoint {
                iterations: 101,
                ..checkpoint()
            },
        ),
    ];
    for (expected, checkpoint) in cases {
        let failure = checkpoint_failure(checkpoint);
        assert_eq!(failure.category, expected);
        let encoded = failure.canonical_encode().unwrap();
        assert_eq!(
            MeshingFailure::canonical_decode(&encoded).unwrap(),
            *failure
        );
    }
}

#[test]
fn stage_control_enforces_wall_time_and_serialization_enforces_artifact_bytes() {
    let mut timed = request();
    timed.resources.maximum_wall_time_ms = 1;
    let mut progress = Progress::default();
    let mut control = MeshingStageControl::new(
        MeshingStageV2::SurfaceMesh,
        0,
        &timed,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    control
        .checkpoint(MeshingStageCheckpoint::default())
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    assert_eq!(
        control.guard().unwrap_err().category,
        MeshingFailureCategory::TimeBudgetExceeded
    );

    let mut bounded = request();
    bounded.resources.maximum_artifact_bytes = 3000;
    let mut fixture = Fixture::with_request(bounded);
    let error = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert_eq!(
        stage_failure(&error).category,
        MeshingFailureCategory::ArtifactBudgetExceeded
    );
}

#[test]
fn serial_stage_rehashes_every_input_before_invoking_the_kernel() {
    let mut fixture = Fixture::new();
    let ValuePayload::Object(root) = &fixture.program.arguments[0] else {
        panic!("fixture input is not externalized")
    };
    fixture
        .store
        .objects
        .insert(root.logical_digest, b"poisoned".to_vec());
    let error = execute_serial_stage(
        &fixture.program,
        &mut fixture.store,
        &SurfaceKernel,
        &NeverCancelled,
        &mut Progress::default(),
        chunk_policy(1024),
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(error, MeshingSerialExecutionError::Bridge(_)));
    assert!(error.to_string().contains("wrong digest"));
}

struct Fixture {
    host: MeshingHostWorkloadV2,
    program: runmat_execution_artifact::ProgramExecutionRequest,
    store: MemoryStore,
}

impl Fixture {
    fn new() -> Self {
        Self::with_request(request())
    }

    fn with_request(request: MeshingRequestV2) -> Self {
        let access = MeshingArtifactAccess {
            authorization_scope: "serial-meshing-run".into(),
            encryption_context: Digest::sha256(b"serial-meshing-encryption-context"),
        };
        let input = input_publication(access.clone());
        let root = root(input.root_output());
        let mut store = MemoryStore::default();
        for object in &input.stage_objects().objects {
            store.write_verified(object).unwrap();
        }
        let root_digest = StableDigest::from_bytes(*root.logical_digest.bytes());
        let identity = MeshingStageIdentityV2 {
            schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
            stage: MeshingStageV2::SurfaceMesh,
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
            prerequisite_artifact_digests: vec![root_digest],
            capability_cohort: Some("native-cohort-v1".into()),
        };
        let workload = MeshingWorkloadRequestV2 {
            schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
            stage: MeshingStageV2::SurfaceMesh,
            stage_identity_digest: identity.canonical_digest().unwrap(),
            partition: MeshingPartitionDescriptorV2 {
                kind: MeshingPartitionKindV2::WholeStage,
                partition_index: 0,
                partition_count: 1,
                entity_range: None,
            },
            input_manifest_digests: vec![root_digest],
            required_capabilities: vec![
                MeshingCapabilityRequirementV2::HostWorkload {
                    abi: "host-v2".into(),
                },
                MeshingCapabilityRequirementV2::MeshingAlgorithm {
                    version: "surface/v2".into(),
                },
                MeshingCapabilityRequirementV2::ElementOrder {
                    order: MeshElementOrderV2::Tet4,
                },
                MeshingCapabilityRequirementV2::DeterministicPlatformCohort {
                    cohort: "native-cohort-v1".into(),
                },
            ],
        };
        let host = MeshingHostWorkloadV2::new(workload, identity, request, access).unwrap();
        let program = host
            .program_request(revision(), std::slice::from_ref(&root))
            .unwrap();
        Self {
            host,
            program,
            store,
        }
    }
}

fn checkpoint() -> MeshingStageCheckpoint {
    MeshingStageCheckpoint {
        completed_work: 1,
        estimated_work: 1,
        ..MeshingStageCheckpoint::default()
    }
}

fn checkpoint_failure(checkpoint: MeshingStageCheckpoint) -> Box<MeshingFailure> {
    let request = request();
    let mut progress = Progress::default();
    let mut control = MeshingStageControl::new(
        MeshingStageV2::SurfaceMesh,
        0,
        &request,
        &NeverCancelled,
        &mut progress,
    )
    .unwrap();
    control
        .checkpoint(MeshingStageCheckpoint::default())
        .unwrap();
    control.checkpoint(checkpoint).unwrap_err()
}

fn stage_failure(error: &MeshingSerialExecutionError) -> &MeshingFailure {
    let MeshingSerialExecutionError::Stage(failure) = error else {
        panic!("expected a typed meshing stage failure: {error}")
    };
    failure
}

fn input_publication(access: MeshingArtifactAccess) -> crate::PreparedMeshingResultPublication {
    let payload = build_chunked_stage_payload(
        &[MeshingChunkStreamV2 {
            media_type: MeshingChunkMediaTypeV2::ExactGeometry,
            schema_version: 2,
            records: vec![b"validated-exact-geometry".to_vec()],
        }],
        chunk_policy(1024),
    )
    .unwrap();
    let (identity, manifest) = build_closed_stage_manifest(
        MeshingStageV2::Healing,
        MeshingStageResultKindV2::WholeStage,
        stable(70),
        stable(71),
        Vec::new(),
        MeshingManifestDispositionV2::ValidatedDependency,
        &payload,
    )
    .unwrap();
    let objects = prepare_stage_objects(
        identity,
        manifest,
        payload.chunks,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    prepare_result_publication(objects, access, ObjectInventoryLimits::default()).unwrap()
}

fn root(payload: &ValuePayload) -> ValueRef {
    let ValuePayload::Object(root) = payload else {
        panic!("publication root is not externalized")
    };
    (**root).clone()
}

fn chunk_policy(maximum_chunk_bytes: u64) -> MeshingChunkPolicyV2 {
    MeshingChunkPolicyV2 {
        maximum_chunk_bytes,
        maximum_records_per_chunk: 10,
        maximum_total_encoded_bytes: 1_000_000,
    }
}

fn request() -> MeshingRequestV2 {
    MeshingRequestV2 {
        schema_version: MESHING_REQUEST_SCHEMA_VERSION,
        element_order: MeshElementOrderV2::Tet4,
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
        metric: MetricFieldRequestV2 {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(0.5).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargetsV2 {
            surface: SurfaceQualityTargetsV2 {
                minimum_metric_angle_degrees: 20.0,
                maximum_physical_aspect_ratio: 10.0,
                maximum_chordal_deviation_m: 1.0e-5,
                maximum_normal_deviation_degrees: 5.0,
            },
            volume: VolumeQualityTargetsV2 {
                maximum_radius_edge_ratio: 2.0,
                minimum_scaled_jacobian: 0.05,
                maximum_metric_edge_length: 1.5,
            },
        },
        resources: MeshingResourceBudgetV2 {
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
        cancellation: CancellationPolicyV2 {
            maximum_checkpoint_latency_ms: 1000,
            maximum_work_units_between_checks: 1000,
        },
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"serial-meshing-graph"),
        Digest::sha256(b"serial-meshing-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"serial-meshing-runtime"),
            Digest::sha256(b"serial-meshing-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn stable(seed: u8) -> StableDigest {
    StableDigest::from_bytes([seed; 32])
}
