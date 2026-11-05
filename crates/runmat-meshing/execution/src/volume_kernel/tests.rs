use runmat_meshing_core::{MeshingFailureCategory, MeshingStageKind};
use runmat_meshing_tetrahedron::cdt::{
    DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
};

use super::{error::map_volume_error, volume_options};

#[test]
fn resolved_request_bounds_every_general_volume_phase() {
    let mut request = crate::serial_tests::request();
    request.limits.maximum_nodes = 10;
    request.limits.maximum_elements = 11;
    request.limits.maximum_recursion_depth = 7;
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

    let segment = options.carving.facet_recovery.segment_recovery;
    assert_eq!(segment.constraints, options.constraints);
    assert_eq!(segment.maximum_steiner_nodes, 10);
    assert_eq!(segment.maximum_split_depth, 7);

    let facet = options.carving.facet_recovery;
    assert_eq!(facet.maximum_cavity_tetrahedra, 11);
    assert_eq!(facet.maximum_cavity_nodes, 10);
    assert_eq!(facet.maximum_cavity_boundary_faces, 11);
    assert_eq!(facet.maximum_cavity_candidate_tetrahedra, 11);
    assert_eq!(facet.maximum_cavity_expansion_rounds, 7);
    assert_eq!(facet.maximum_cavity_steiner_nodes, 10);

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
    assert!(options.refinement.maximum_insertions <= 10);
}

#[test]
fn native_volume_checkpoint_batches_preserve_tighter_kernel_cadences() {
    let mut request = crate::serial_tests::request();
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

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The governed request projects its exact declared search envelope at small,
//   standalone-default and million-operation sizes. Shared cumulative stage accounting still bounds
//   every strategy. This contract check establishes neither quality convergence nor original mesh
//   acceptance.
#[test]
fn volume_algorithm_bounds_do_not_depend_on_execution_ceilings() {
    // `volume_options` sees only the request: its size limits and the algorithms' own bounds.
    // Execution ceilings are enforced by the stage control and cannot change the mesh.
    let request = crate::serial_tests::request();
    let options = volume_options(&request);
    assert_eq!(
        options.sliver.maximum_candidate_evaluations_per_pass,
        super::options::VOLUME_ALGORITHM_SEARCH
    );
    assert_eq!(options, volume_options(&request));
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

#[test]
fn volume_accounting_failure_retains_original_cause() {
    let cause = runmat_simulation_usage::LedgerError::CountOverflow;
    let failure = map_volume_error(DelaunayVolumeMeshError {
        stage: DelaunayVolumeMeshStage::Quality,
        kind: DelaunayVolumeMeshErrorKind::Accounting(cause),
        reason: "operation count overflow".into(),
    });
    assert_eq!(
        failure.category,
        MeshingFailureCategory::NumericalAccounting
    );
    assert_eq!(failure.accounting_error, Some(cause));
    failure
        .validate()
        .expect("typed accounting failure contract");
}
