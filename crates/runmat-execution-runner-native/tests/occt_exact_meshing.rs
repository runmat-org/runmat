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
    ElementOrder, MeshingChunkPolicy, MeshingDomainModel, MeshingQualityTargets, MeshingRequest,
    MeshingResourceBudget, MetricCombinationRule, MetricFieldRequest, MetricTensor3,
    NeverCancelled, PlatformBuildIdentity, RegionMaterialAssignment, StableDigest,
    SurfaceQualityTargets, VolumeQualityTargets, MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_domain_model_input, prepare_domain_model_objects,
    prepare_exact_geometry_input, prepare_exact_geometry_objects, ExactMeshingDagRun,
    MeshingArtifactAccess, MeshingRunEvidenceContext, NoopMeshingProgress,
    SerialExactMeshingExecutor,
};

const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");

#[path = "occt_exact_meshing/differential.rs"]
mod differential;

#[test]
fn occt_box_executes_the_complete_serial_exact_dag() {
    let (_, result) = execute_box();
    result.evidence.validate(&result.artifact).unwrap();
    assert!(!result.artifact.topology.volume_elements.is_empty());
}

fn execute_box() -> (
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
    let region_id = imported.topology.regions[0].id.clone();
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
    let domain_objects = prepare_domain_model_objects(
        MeshingDomainModel {
            schema_version: MESHING_DOMAIN_MODEL_SCHEMA_VERSION,
            region_materials: vec![RegionMaterialAssignment {
                region_id,
                material_id: "steel".into(),
            }],
            contact_ids: Vec::new(),
        },
        limits,
    )
    .unwrap();
    let domain = prepare_domain_model_input(domain_objects, access.clone(), limits).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store =
        FilesystemObjectStore::open(directory.path().join("objects"), 512 << 20).unwrap();
    for object in geometry
        .geometry_objects()
        .objects
        .iter()
        .chain(&domain.domain_model_objects().objects)
    {
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
            domain_model: &domain,
            request: request(document.tolerance),
            artifact_access: access,
            capability_cohort: Some("occt-native-corpus".into()),
            program_revision: revision(),
            preferred_edges_per_partition: 8,
            preferred_faces_per_partition: 8,
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
