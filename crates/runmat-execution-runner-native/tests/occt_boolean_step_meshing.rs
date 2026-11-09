#![cfg(feature = "occt-native")]
//! Exact meshing of STEP parts that a CAD kernel built with booleans. OCCT boolean results on
//! B-spline faces carry pcurves that meet each vertex within the vertex's 3D tolerance (about
//! 1e-7 m), so incident pcurves disagree in UV by that tolerance mapped through the face's
//! parametric scale. Curve meshing admits each endpoint against the vertex tolerance in 3D.
//!
//! Fixtures in `fixtures/boolean_step/` are runmat-cad exports (OCCT 7.8) of a camera drone's
//! arm and ESC shell:
//! - `beam_loft_only.step`: the arm's lofted beam alone (closed-spline loft).
//! - `arm_box_beam.step`: knuckle and pad cylinders joined by a planar box beam.
//! - `arm_only.step`: the knuckle and pad cylinders unioned with the lofted beam.
//! - `arm_study.step`: the arm and the motor pod seated in a pocket over the arm's pad, as two
//!   touching solids.
//! - `esc_thermal.step`: a lofted lower shell trimmed by a plane and cut by an inner loft, with
//!   bosses joined to it, and the ESC board touching it.

use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, FilesystemObjectStore};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ArtifactAccess;
use runmat_execution_runner_native::native_meshing_kernel_dispatcher;
use runmat_geometry_io::{
    import_exact_cad, ExactCadImportOptions, GeometryFormat, GeometryImportContext,
};
use runmat_meshing_core::{
    CacheAdmissionDecision, ElementOrder, MeshingChunkPolicy, MeshingRequestSettings,
    NeverCancelled, PlatformBuildIdentity, StableDigest,
};
use runmat_meshing_execution::{
    execute_exact_meshing_dag, prepare_exact_geometry_input, prepare_exact_geometry_objects,
    ExactMeshingDagRun, ExactMeshingDagRunResult, MeshingRunEvidenceContext, NoopMeshingProgress,
    SerialExactMeshingExecutor,
};

const BEAM_LOFT_ONLY: &[u8] = include_bytes!("fixtures/boolean_step/beam_loft_only.step");
const ARM_BOX_BEAM: &[u8] = include_bytes!("fixtures/boolean_step/arm_box_beam.step");
const ARM_ONLY: &[u8] = include_bytes!("fixtures/boolean_step/arm_only.step");
const ARM_STUDY: &[u8] = include_bytes!("fixtures/boolean_step/arm_study.step");
const ESC_THERMAL: &[u8] = include_bytes!("fixtures/boolean_step/esc_thermal.step");
const HEATSINK: &[u8] =
    include_bytes!("../../../docs/simulation/examples/geometry/electronics-heatsink.step");

#[test]
fn a_lofted_beam_alone_meshes() {
    assert_meshes("beam_loft_only.step", BEAM_LOFT_ONLY, false, 3.0e-3, 1);
}

#[test]
fn cylinders_joined_by_a_planar_beam_mesh() {
    assert_meshes("arm_box_beam.step", ARM_BOX_BEAM, false, 3.0e-3, 1);
}

#[test]
fn cylinders_unioned_with_a_lofted_beam_mesh() {
    assert_meshes("arm_only.step", ARM_ONLY, false, 3.0e-3, 1);
}

#[test]
fn the_arm_and_its_motor_pod_mesh_with_a_shared_interface() {
    assert_meshes("arm_study.step", ARM_STUDY, true, 3.0e-3, 2);
}

/// The docs heat-sink example (KiCad library model): 0.25 mm chamfers along every fin edge
/// and 2.4 um sliver faces at the fin roots. Facet recovery must recover the sliver needles.
#[test]
#[ignore = "over 30 min in release (about 330k surface triangles); run with --ignored when changing surface sizing or facet recovery"]
fn the_docs_heat_sink_meshes() {
    assert_meshes("electronics-heatsink.step", HEATSINK, false, 10.0e-3, 1);
}

#[test]
#[ignore = "tetrahedralization facet recovery: constrained Delaunay Steiner insertion leaves a non-Delaunay neighbor on the y = 31.5 mm trim plane"]
fn a_trimmed_and_cut_lofted_shell_with_its_board_meshes_with_a_shared_interface() {
    assert_meshes("esc_thermal.step", ESC_THERMAL, true, 2.5e-3, 2);
}

fn assert_meshes(
    source_name: &str,
    source_bytes: &[u8],
    shared_interfaces: bool,
    target_size_m: f64,
    solids: usize,
) -> usize {
    let result = mesh_step(source_name, source_bytes, shared_interfaces, target_size_m)
        .unwrap_or_else(|error| panic!("{source_name} must mesh: {error}"));
    let topology = &result.artifact.topology;
    assert_eq!(topology.regions.len(), solids, "{source_name} regions");
    assert!(
        !topology.volume_elements.is_empty(),
        "{source_name} volume elements"
    );
    eprintln!(
        "{source_name}: {} nodes, {} volume elements",
        topology.nodes.len(),
        topology.volume_elements.len()
    );
    topology.volume_elements.len()
}

fn mesh_step(
    source_name: &str,
    source_bytes: &[u8],
    shared_interfaces: bool,
    target_size_m: f64,
) -> Result<ExactMeshingDagRunResult, String> {
    let mut options = ExactCadImportOptions::default();
    options.analysis.healing.shared_interfaces = shared_interfaces;
    let imported = import_exact_cad(
        source_name,
        source_bytes,
        GeometryFormat::Step,
        &options,
        &GeometryImportContext::new(),
    )
    .map_err(|error| error.to_string())?;
    let document = imported.geometry_document().map_err(|e| e.to_string())?;
    let limits = ObjectInventoryLimits::default();
    let access = ArtifactAccess {
        authorization_scope: "occt-boolean-step-corpus".into(),
        encryption_context: Digest::sha256(b"occt-boolean-step-corpus"),
    };
    let geometry_objects = prepare_exact_geometry_objects(
        document.clone(),
        imported.topology,
        imported.evaluators,
        Some(imported.representation),
        imported.healing_report,
        limits,
    )
    .map_err(|e| e.to_string())?;
    let geometry = prepare_exact_geometry_input(geometry_objects, access.clone(), limits)
        .map_err(|e| e.to_string())?;
    let directory = tempfile::tempdir().unwrap();
    let mut store = FilesystemObjectStore::open(directory.path().join("objects"), 2 << 30).unwrap();
    for object in &geometry.geometry_objects().objects {
        store.write_verified(object).unwrap();
    }
    let request = runmat_meshing_core::resolve_meshing_request(
        document.tolerance,
        MeshingRequestSettings {
            element_order: ElementOrder::Tet4,
            deterministic_seed: 7,
            target_edge_length_m: runmat_types::physical::Length::new(target_size_m),
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            ..MeshingRequestSettings::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let mut progress = NoopMeshingProgress;
    let mut executor = SerialExactMeshingExecutor {
        store: &mut store,
        kernel: &native_meshing_kernel_dispatcher(),
        cancellation: &NeverCancelled,
        progress: &mut progress,
        chunk_policy: MeshingChunkPolicy {
            maximum_chunk_bytes: 16 << 20,
            maximum_records_per_chunk: 65_536,
            maximum_total_encoded_bytes: 2 << 30,
        },
        inventory_limits: limits,
    };
    execute_exact_meshing_dag(
        ExactMeshingDagRun {
            geometry: &geometry,
            ceilings: runmat_meshing_core::MeshingExecutionCeilings::PORTABLE_DEFAULT,
            request,
            artifact_access: access,
            capability_cohort: Some("occt-native-corpus".into()),
            program_revision: revision(),
            preferred_edges_per_partition: 64,
            preferred_faces_per_partition: 16,
            inventory_limits: limits,
            evidence: MeshingRunEvidenceContext {
                platform: PlatformBuildIdentity {
                    capability_cohort: "occt-native-corpus".into(),
                    target_triple: std::env::consts::ARCH.into(),
                    build_digest: StableDigest::from_bytes([72; 32]),
                    exact_kernel_abi: document.source.kernel_version.clone(),
                },
                sizing: Vec::new(),
                cache_admission: CacheAdmissionDecision::Admitted,
            },
            named_regions: Default::default(),
        },
        &mut executor,
    )
    .map_err(|error| error.to_string())
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256(b"occt-boolean-step-meshing-graph"),
        Digest::sha256(b"occt-boolean-step-meshing-source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"occt-boolean-step-meshing-runtime"),
            Digest::sha256(b"occt-boolean-step-meshing-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}
