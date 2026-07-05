use super::*;

#[test]
fn source_edge_split_recovery_evaluates_refill_after_splitting_incident_tetrahedra() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let source_tetrahedra = vec![candidate_tetrahedron(1, [0, 1, 2, 3], 1.0 / 6.0, &["body"])];

    let recovery = recover_constrained_cavity_source_edge_by_split_refill(
        &cavity,
        &nodes,
        &nodes,
        &source_tetrahedra,
        [1, 0],
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.0,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("source-edge split recovery should evaluate refill candidates");

    assert_eq!(recovery.split.split_node.node_id, 4);
    assert_eq!(recovery.split.split_node.coordinates_m, [0.5, 0.0, 0.0]);
    assert_eq!(recovery.split.source_tetrahedra.len(), 2);
    assert_eq!(recovery.split.cavity.removed_tetrahedron_ids, vec![2, 3]);
    assert!(recovery.refill_evaluation.refill.is_some());
    assert!(recovery.refill_evaluation.rejected_by_reason.is_empty());
    assert_eq!(cavity.removed_tetrahedron_ids, vec![1]);
    assert_eq!(source_tetrahedra[0].tetrahedron_id, 1);
}

#[test]
fn source_edge_split_recovery_rejects_without_mutating_inputs_when_edge_is_not_on_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let source_tetrahedra = vec![candidate_tetrahedron(1, [0, 1, 2, 3], 1.0 / 6.0, &["body"])];

    let err = recover_constrained_cavity_source_edge_by_split_refill(
        &cavity,
        &nodes,
        &nodes,
        &source_tetrahedra,
        [0, 99],
        ConstrainedCavityRefillOptions::default(),
    )
    .expect_err("non-boundary source edge should not split");

    assert_eq!(
        err,
        ConstrainedCavitySourceEdgeSplitError::MissingBoundaryNode { node_id: 99 }
    );
    assert_eq!(cavity.boundary_faces.len(), 4);
    assert_eq!(source_tetrahedra[0].tetrahedron_id, 1);
}
