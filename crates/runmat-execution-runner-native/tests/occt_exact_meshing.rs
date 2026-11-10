#![cfg(feature = "occt-native")]

use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::CacheExport;
use runmat_execution_artifact::cache::FilesystemObjectStore;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::native_meshing_kernel_dispatcher;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
};
use runmat_meshing_core::{
    AlgorithmVersionSet, CacheAdmissionDecision, CancellationPolicy, CurveQualityTargets,
    ElementOrder, MeshingChunkPolicy, MeshingQualityTargets, MeshingRequest, MeshingSizeLimits,
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled,
    PlatformBuildIdentity, StableDigest, SurfaceQualityTargets, VolumeQualityTargets,
    MESHING_REQUEST_SCHEMA_VERSION,
};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_exact_geometry_input, prepare_exact_geometry_objects,
    ExactMeshingDagRun, MeshingRunEvidenceContext, NoopMeshingProgress, SerialExactMeshingExecutor,
};

const BOX: &[u8] = include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep");
const DISCONNECTED_SOLIDS: &[u8] =
    include_bytes!("../../runmat-geometry/io/tests/fixtures/disconnected_solids.brep");
// Schema 9 identity includes the typed-unit contracts. The differential test
// independently checks this box's geometry, volume and linear-field energy.
const EXPECTED_BOX_ARTIFACT_DIGEST: &str =
    "ca159734b541dee8e9935960c60361fbcf67b1cddc19880c0bd265921254d827";

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

#[test]
fn disconnected_cad_solids_retain_distinct_region_identity_through_meshing() {
    let (topology, result) =
        execute_fixture_with_layout("disconnected_solids.brep", DISCONNECTED_SOLIDS, 8, 8);
    let source_regions = topology
        .regions
        .iter()
        .map(|region| region.id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let meshed_regions = result
        .artifact
        .topology
        .regions
        .iter()
        .map(|region| region.region_id.clone())
        .collect::<std::collections::BTreeSet<_>>();

    assert_eq!(source_regions.len(), 2);
    assert_eq!(source_regions, meshed_regions);
    assert!(result
        .artifact
        .topology
        .volume_elements
        .iter()
        .all(|element| {
            result
                .artifact
                .topology
                .regions
                .iter()
                .any(|region| region.region_id == element.region_id)
        }));

    // Model regions name each CAD solid by its canonical persistent selector; the solver matches
    // that selector against the mesh provenance directly.
    let selectors = topology
        .regions
        .iter()
        .map(|region| region.id.analysis_selection_id().unwrap())
        .collect::<Vec<_>>();
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    let mut second_material = model.materials[0].clone();
    second_material.material_id = "second-solid-material".into();
    second_material.name = "Second solid material".into();
    second_material.mass_density_kg_per_m3 = (second_material
        .mass_density_kg_per_m3
        .map(|quantity| quantity.canonical_value())
        .map(|density| density * 0.5))
    .map(runmat_types::physical::MassDensity::new);
    model.materials.push(second_material);
    model.material_assignments = vec![
        material_assignment(&selectors[0], &model.materials[0].material_id),
        material_assignment(&selectors[1], &model.materials[1].material_id),
    ];
    let inventory =
        runmat_analysis_fea::properties::compile_solid_property_map(&model, &result.artifact)
            .expect("two selected CAD solids should compile their material assignments");
    runmat_analysis_fea::assembly::solver_solid::assemble_solver_solid_operators(
        &result.artifact,
        &inventory,
    )
    .expect("two selected CAD solids should receive exactly one material each");

    // A legacy importer ordinal is not an identity and selects nothing.
    let mut legacy = model.clone();
    legacy.material_assignments[0] =
        material_assignment("solid_000001", &legacy.materials[0].material_id);
    let error =
        runmat_analysis_fea::properties::compile_solid_property_map(&legacy, &result.artifact)
            .expect_err("a legacy solid_000001 selector must not match a meshed CAD solid");
    assert!(
        error.to_string().contains("solid_000001")
            || matches!(
                error,
                runmat_analysis_fea::properties::PropertyCompilationError::InvalidCoverage { .. }
            )
    );

    model.material_assignments.pop();
    let error =
        runmat_analysis_fea::properties::compile_solid_property_map(&model, &result.artifact)
            .expect_err("unassigned CAD solids must fail during property compilation");
    assert!(matches!(
        error,
        runmat_analysis_fea::properties::PropertyCompilationError::InvalidCoverage {
            role: "solid",
            assignment_count: 0,
            ..
        }
    ));
}

fn material_assignment(
    region: &str,
    material_id: &str,
) -> runmat_analysis_core::MaterialAssignment {
    runmat_analysis_core::MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::solid(region),
        expected_material_id: material_id.into(),
        assigned_material_id: material_id.into(),
        confidence: runmat_analysis_core::EvidenceConfidence::Verified,
    }
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
    execute_fixture_with_layout(
        "box.brep",
        BOX,
        preferred_edges_per_partition,
        preferred_faces_per_partition,
    )
}

fn execute_fixture_with_layout(
    source_name: &str,
    source_bytes: &[u8],
    preferred_edges_per_partition: u32,
    preferred_faces_per_partition: u32,
) -> (
    runmat_geometry_core::ExactBRepTopology,
    runmat_meshing_execution::ExactMeshingDagRunResult,
) {
    let imported = import_exact_cad(
        source_name,
        source_bytes,
        GeometryFormat::Brep,
        &ExactCadImportOptions {
            source_units: Some(runmat_geometry_core::LengthUnit::Meter),
            ..ExactCadImportOptions::default()
        },
        &GeometryImportContext::new(),
    )
    .unwrap();
    let topology = imported.topology.clone();
    let document = imported.geometry_document().unwrap();
    let limits = ObjectInventoryLimits::default();
    let access = ArtifactAccess {
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
            ceilings: runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
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
            named_regions: Default::default(),
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
        tolerance,
        metric: MetricFieldRequest {
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(10.0).unwrap(),
            maximum_grading_ratio: 1.3,
            contributions: Vec::new(),
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            cells_across_gap: None,
        },
        quality: MeshingQualityTargets {
            curve: CurveQualityTargets {
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.1),
                maximum_tangent_change_degrees: 180.0,
                minimum_metric_edge_length: 0.01,
                maximum_metric_edge_length: 10.0,
            },
            surface: SurfaceQualityTargets {
                minimum_metric_edge_length: 0.01,
                minimum_metric_angle_degrees: 0.1,
                maximum_physical_aspect_ratio: 1_000.0,
                maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.1),
                maximum_normal_deviation_degrees: 180.0,
            },
            volume: VolumeQualityTargets {
                maximum_radius_edge_ratio: 10.0,
                minimum_scaled_jacobian: 0.001,
                maximum_metric_edge_length: 10.0,
            },
        },
        limits: MeshingSizeLimits {
            maximum_nodes: 10_000,
            maximum_elements: 10_000,
            maximum_recursion_depth: 128,
        },
        cancellation: CancellationPolicy {
            maximum_checkpoint_latency_ms: 10_000,
            maximum_work_units_between_checks: 1_000_000,
        },
        size_controls: Vec::new(),
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
