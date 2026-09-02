use runmat_meshing_core::{MeshingFailureCategory, MeshingStageKind};
use runmat_meshing_tetrahedron::cdt::{
    DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
};

use super::{error::map_volume_error, volume_options};

#[test]
fn resolved_request_bounds_every_general_volume_phase() {
    let mut request = crate::serial_tests::request();
    request.resources.maximum_nodes = 10;
    request.resources.maximum_elements = 11;
    request.resources.maximum_search_work = 12;
    request.resources.maximum_recursion_depth = 7;
    request.resources.maximum_iterations = 8;
    request.cancellation.maximum_work_units_between_checks = 5;
    request.quality.volume.maximum_metric_edge_length = 1.25;
    request.quality.volume.maximum_radius_edge_ratio = 3.5;
    request.quality.volume.minimum_scaled_jacobian = 0.025;

    let options = volume_options(&request);
    assert_eq!(options.constraints.maximum_nodes, 10);
    assert_eq!(options.constraints.maximum_segments, 11);
    assert_eq!(options.constraints.maximum_facets, 11);
    assert_eq!(options.constraints.cancellation_check_interval, 5);

    let insertion = options.carving.facet_recovery.segment_recovery.insertion;
    assert_eq!(insertion.topology.maximum_nodes, 10);
    assert_eq!(insertion.topology.maximum_tetrahedra, 11);
    assert_eq!(insertion.maximum_protected_faces, 11);
    assert_eq!(insertion.maximum_cavity_tetrahedra, 11);
    assert_eq!(insertion.maximum_cavity_boundary_faces, 11);
    assert_eq!(insertion.maximum_predicate_evaluations, 12);

    let segment = options.carving.facet_recovery.segment_recovery;
    assert_eq!(segment.constraints, options.constraints);
    assert_eq!(segment.maximum_steiner_nodes, 10);
    assert_eq!(segment.maximum_recovery_steps, 12);
    assert_eq!(segment.maximum_search_steps, 12);
    assert_eq!(segment.maximum_flip_attempts, 12);
    assert_eq!(segment.maximum_split_depth, 7);
    assert_eq!(segment.maximum_recovery_passes, 8);

    let facet = options.carving.facet_recovery;
    assert_eq!(facet.maximum_search_steps, 12);
    assert_eq!(facet.maximum_flip_attempts, 12);
    assert_eq!(facet.maximum_support_steps, 12);
    assert_eq!(facet.maximum_cavity_steps, 12);
    assert_eq!(facet.maximum_cavity_tetrahedra, 11);
    assert_eq!(facet.maximum_cavity_nodes, 10);
    assert_eq!(facet.maximum_cavity_boundary_faces, 11);
    assert_eq!(facet.maximum_cavity_candidate_tetrahedra, 11);
    assert_eq!(facet.maximum_cavity_expansion_rounds, 7);
    assert_eq!(facet.maximum_cavity_steiner_nodes, 10);
    assert!(facet.maximum_cavity_candidate_evaluations <= 12);
    assert!(facet.maximum_cavity_exact_cover_attempts <= 12);
    assert!(facet.maximum_cavity_steiner_candidates <= 12);
    assert!(facet.maximum_cavity_steiner_candidate_evaluations_per_round <= 12);
    assert_eq!(options.carving.maximum_flood_steps, 12);

    assert_eq!(options.provenance.maximum_node_bindings, 10);
    assert_eq!(options.provenance.maximum_segment_bindings, 11);
    assert_eq!(options.provenance.maximum_facet_bindings, 11);
    assert_eq!(options.quality.maximum_nodes, 10);
    assert_eq!(options.quality.maximum_tetrahedra, 11);
    assert_eq!(options.quality.maximum_metric_edge_length, 1.25);
    assert_eq!(options.quality.maximum_radius_edge_ratio, 3.5);
    assert_eq!(options.quality.minimum_metric_scaled_jacobian, 0.025);
    assert_eq!(options.quality.provenance, options.provenance);
    assert_eq!(options.refinement.step.insertion, insertion);
    assert_eq!(options.sliver.insertion, insertion);
    assert!(options.refinement.maximum_insertions <= 8);
    assert!(options.sliver.maximum_passes <= 8);
    assert!(options.sliver.maximum_candidate_evaluations_per_pass <= 12);
    assert!(options.sliver.maximum_compound_reconnections_per_insertion <= 12);
}

#[test]
fn native_volume_checkpoint_batches_preserve_tighter_kernel_cadences() {
    let mut request = crate::serial_tests::request();
    request.resources.maximum_search_work = 1_000_000;
    request.cancellation.maximum_work_units_between_checks = 100_000;

    let options = volume_options(&request);
    assert_eq!(options.constraints.cancellation_check_interval, 32);
    assert_eq!(
        options
            .carving
            .facet_recovery
            .segment_recovery
            .insertion
            .topology
            .cancellation_check_interval,
        1
    );
    assert_eq!(options.provenance.cancellation_check_interval, 32);
    assert_eq!(options.quality.cancellation_check_interval, 32);
    assert_eq!(
        options
            .refinement
            .step
            .candidate
            .cancellation_check_interval,
        1
    );
    assert_eq!(options.sliver.cancellation_check_interval, 32);
}

#[test]
fn general_volume_errors_keep_typed_stage_categories() {
    for (stage, expected) in [
        (
            DelaunayVolumeMeshStage::PointSet,
            MeshingFailureCategory::NodeBudgetExceeded,
        ),
        (
            DelaunayVolumeMeshStage::FacetRecovery,
            MeshingFailureCategory::SearchWorkBudgetExceeded,
        ),
        (
            DelaunayVolumeMeshStage::Quality,
            MeshingFailureCategory::ElementBudgetExceeded,
        ),
    ] {
        let failure = map_volume_error(DelaunayVolumeMeshError {
            stage,
            kind: DelaunayVolumeMeshErrorKind::ResourceLimit,
            reason: "bounded test failure".into(),
        });
        assert_eq!(failure.stage, MeshingStageKind::Tetrahedralization);
        assert_eq!(failure.category, expected);
    }
}
