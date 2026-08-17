use std::collections::BTreeMap;
use std::path::Path;

use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_meshing_core::{
    AlgorithmVersionSet, CancellationPolicyV2, CanonicalMeshingContract, GeometryRevisionRef,
    GeometryTolerancePolicy, MeshElementOrderV2, MeshingCapabilityRequirementV2,
    MeshingChunkMediaTypeV2, MeshingChunkPolicyV2, MeshingChunkStreamV2, MeshingFailure,
    MeshingPartitionDescriptorV2, MeshingPartitionKindV2, MeshingQualityTargetsV2,
    MeshingRequestV2, MeshingResourceBudgetV2, MeshingStageIdentityV2, MeshingStageV2,
    MeshingWorkloadRequestV2, MetricCombinationRule, MetricFieldRequestV2, MetricTensor3,
    NeverCancelled, StableDigest, SurfaceQualityTargetsV2, VolumeQualityTargetsV2,
    MESHING_IDENTITY_SCHEMA_VERSION, MESHING_REQUEST_SCHEMA_VERSION,
    MESHING_WORKLOAD_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    import_result_publication, MeshingArtifactAccess, MeshingHostWorkloadV2,
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel, NoopMeshingProgress,
    ValidatedMeshingStageOutput,
};
use runmat_process_host::environment::EnvironmentPolicy;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use runmat_process_host::HostCommand;
use tokio::io::BufReader;

use runmat_execution_runner_native::{
    execute_meshing_program_request, run_meshing_worker_stdio, NativeMeshingHostLimits,
    NativeObjectStore,
};

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
            streams: vec![MeshingChunkStreamV2 {
                media_type: MeshingChunkMediaTypeV2::ExactGeometry,
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

fn main() {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    if arguments
        .get(1)
        .is_some_and(|argument| argument == "--child")
    {
        let root = arguments.get(2).expect("child object-store root");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_meshing_worker_stdio(
                &AdmissionKernel,
                Path::new(root),
                limits(),
            ))
            .unwrap();
        return;
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(parent());
}

async fn parent() {
    let directory = tempfile::tempdir().unwrap();
    let store_root = directory.path().join("objects");
    let store = NativeObjectStore::open(&store_root, limits().inventory.max_object_bytes).unwrap();
    let (host, request) = fixture();

    let mut command = HostCommand::new(std::env::current_exe().unwrap());
    command.arguments = vec!["--child".into(), store_root.to_string_lossy().into_owned()];
    command.environment_policy = EnvironmentPolicy::Clear;
    let mut child = command.spawn().await.unwrap();
    let stderr = child.captured_stderr();
    let stdio = child.take_stdio().unwrap();
    let mut reader = BufReader::new(stdio.stdout);
    let mut writer = stdio.stdin;
    let frame_limits = FrameLimits {
        max_message_bytes: limits().max_message_bytes,
    };
    write_payload(
        &mut writer,
        &serde_json::to_vec(&request).unwrap(),
        frame_limits,
    )
    .await
    .unwrap();
    let payload = read_payload(&mut reader, frame_limits)
        .await
        .unwrap_or_else(|error| panic!("child frame failed: {error}; stderr: {}", stderr.text()));
    let response: ProgramExecutionResponse = serde_json::from_slice(&payload).unwrap();
    response.validate_against(&request).unwrap();
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        panic!("native meshing child did not return an externalized success: {response:?}")
    };
    assert!(serde_json::to_vec(&outputs).unwrap().len() < 4096);
    assert!(result_objects.len() >= 3);
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        panic!("native meshing child returned a non-object root")
    };
    let imported =
        import_result_publication(&store, root, host.artifact_access, limits().inventory).unwrap();
    assert_eq!(imported.result_objects(), result_objects);
    let exit = child.wait().await.unwrap();
    assert!(exit.success, "child failed: {}", stderr.text());

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
}

fn fixture() -> (
    MeshingHostWorkloadV2,
    runmat_execution_artifact::ProgramExecutionRequest,
) {
    let access = MeshingArtifactAccess {
        authorization_scope: "native-meshing-run".into(),
        encryption_context: Digest::sha256(b"native-meshing-encryption-context"),
    };
    let request = request();
    let identity = MeshingStageIdentityV2 {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: MeshingStageV2::GeometryAdmission,
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
        prerequisite_artifact_digests: Vec::new(),
        capability_cohort: Some("native-cohort-v1".into()),
    };
    let workload = MeshingWorkloadRequestV2 {
        schema_version: MESHING_WORKLOAD_SCHEMA_VERSION,
        stage: MeshingStageV2::GeometryAdmission,
        stage_identity_digest: identity.canonical_digest().unwrap(),
        partition: MeshingPartitionDescriptorV2 {
            kind: MeshingPartitionKindV2::WholeStage,
            partition_index: 0,
            partition_count: 1,
            entity_range: None,
        },
        input_manifest_digests: Vec::new(),
        required_capabilities: vec![
            MeshingCapabilityRequirementV2::HostWorkload {
                abi: "host-v2".into(),
            },
            MeshingCapabilityRequirementV2::ExactCadKernel {
                abi: "occt-v1".into(),
            },
            MeshingCapabilityRequirementV2::MeshingAlgorithm {
                version: "geometry/v2".into(),
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
    let program = host.program_request(revision(), &[]).unwrap();
    (host, program)
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
        chunk_policy: MeshingChunkPolicyV2 {
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
