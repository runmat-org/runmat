#![cfg(feature = "occt-native")]

use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_runner_native::native_meshing_kernel_dispatcher;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
};
use runmat_meshing_core::{
    AlgorithmVersionSet, CacheAdmissionDecision, CancellationPolicy, CurveQualityTargets,
    ElementOrder, MeshingChunkPolicy, MeshingQualityTargets, MeshingRequest, MeshingResourceBudget,
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled,
    PlatformBuildIdentity, StableDigest, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_exact_geometry_input, prepare_exact_geometry_objects,
    ExactMeshingDagRun, MeshingArtifactAccess, MeshingRunEvidenceContext, NoopMeshingProgress,
    SerialExactMeshingExecutor,
};

const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");
const EXPECTED_BOX_ARTIFACT_DIGEST: &str =
    "15477bc5436facd3e152c1fd53be94d959222808a3b608f9a69dea8d4ca7e7b6";

#[path = "occt_exact_meshing/differential.rs"]
mod differential;

#[test]
fn occt_box_executes_the_complete_serial_exact_dag() {
    let (_, result) = execute_box();
    result.evidence.validate(&result.artifact).unwrap();
    assert!(!result.artifact.topology.volume_elements.is_empty());
    assert_eq!(
        digest_hex(result.artifact.canonical_digest.bytes()),
        EXPECTED_BOX_ARTIFACT_DIGEST
    );
}

#[test]
fn occt_box_artifact_is_independent_of_legal_partition_layout() {
    let (_, fine) = execute_box_with_layout(1, 1);
    let (_, coarse) = execute_box_with_layout(32, 32);

    assert_eq!(fine.artifact.topology, coarse.artifact.topology);
    assert_eq!(
        fine.artifact.canonical_digest,
        coarse.artifact.canonical_digest
    );
}

fn execute_box() -> (
    runmat_geometry_core::ExactBRepTopology,
    runmat_meshing_execution::ExactMeshingDagRunResult,
) {
    execute_box_with_layout(8, 8)
}

fn execute_box_with_layout(
    preferred_edges_per_partition: u32,
    preferred_faces_per_partition: u32,
) -> (
    runmat_geometry_core::ExactBRepTopology,
    runmat_meshing_execution::ExactMeshingDagRunResult,
) {
    let imported = import_exact_cad(
        "box.brep",
        BOX,
        GeometryFormat::Brep,
        &ExactCadImportOptions::default(),
        &GeometryImportContext::new(),
    )
    .unwrap();
    let topology = imported.topology.clone();
    let document = imported.geometry_document().unwrap();
    let limits = ObjectInventoryLimits::default();
    let access = MeshingArtifactAccess {
        authorization_scope: "occt-box-serial-corpus".into(),
        encryption_context: Digest::sha256(b"occt-box-serial-corpus"),
    };
    let geometry_objects = prepare_exact_geometry_objects(
        document.clone(),
        imported.topology,
        imported.evaluators,
        Some(imported.representation),
        imported.healing_report,
        limits,
    )
    .unwrap();
    let geometry = prepare_exact_geometry_input(geometry_objects, access.clone(), limits).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store =
        FilesystemObjectStore::open(directory.path().join("objects"), 512 << 20).unwrap();
    for object in &geometry.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
    let mut progress = NoopMeshingProgress;
    let mut executor = SerialExactMeshingExecutor {
        store: &mut store,
        kernel: &native_meshing_kernel_dispatcher(),
        cancellation: &NeverCancelled,
        progress: &mut progress,
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 16 << 20,
            maximum_records_per_chunk: 4096,
            maximum_total_encoded_bytes: 512 << 20,
        },
        inventory_limits: limits,
    };
    let result = execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            request: request(document.tolerance),
            artifact_access: access,
            capability_cohort: Some("occt-native-corpus".into()),
            program_revision: revision(),
            preferred_edges_per_partition,
            preferred_faces_per_partition,
            inventory_limits: limits,
            evidence: MeshingRunEvidenceContext {
                platform: PlatformBuildIdentity {
                    capability_cohort: "occt-native-corpus".into(),
                    target_triple: std::env::consts::ARCH.into(),
                    build_digest: stable(72),
                    exact_kernel_abi: Some(document.source.kernel_version.unwrap()),
                },
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
        },
        &mut executor,
    )
    .unwrap();
    (topology, result)
}

fn request(tolerance: runmat_geometry_core::GeometryTolerancePolicy) -> MeshingRequest {
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
        tolerance,
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(10.0).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: 0.1,
                maximum_tangent_change_degrees: 180.0,
                minimum_metric_edge_length: 0.01,
                maximum_metric_edge_length: 10.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_angle_degrees: 0.1,
                maximum_physical_aspect_ratio: 1_000.0,
                maximum_chordal_deviation_m: 0.1,
                maximum_normal_deviation_degrees: 180.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.001,
                maximum_metric_edge_length: 10.0,
            },
        },
        resources: MeshingResourceBudget {
            maximum_nodes: 10_000,
            maximum_elements: 10_000,
            maximum_memory_bytes: 512 << 20,
            maximum_scratch_bytes: 512 << 20,
            maximum_wall_time_ms: 60_000,
            maximum_artifact_bytes: 512 << 20,
            maximum_search_work: 10_000_000,
            maximum_recursion_depth: 128,
            maximum_iterations: 10_000_000,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 10_000,
            maximum_work_units_between_checks: 1_000_000,
        },
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"occt-box-meshing-graph"),
        Digest::sha256(b"occt-box-meshing-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"occt-box-meshing-runtime"),
            Digest::sha256(b"occt-box-meshing-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn stable(seed: u8) -> StableDigest {
    StableDigest::from_bytes([seed; 32])
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
