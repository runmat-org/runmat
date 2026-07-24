use super::*;

#[test]
fn boundary_edge_star_recovery_reports_added_tetrahedra() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 10, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        [1, 0],
        &[],
    )
    .expect("boundary edge-star recovery should evaluate");

    assert_eq!(recovery.attempted_boundary_faces, Vec::<[u32; 3]>::new());
    assert_eq!(
        recovery.recovered_edge,
        Some(ConstrainedCavityBoundaryEdgeRecoveryStep {
            node_ids: [0, 1],
            added_tetrahedron_ids: vec![11, 12, 13],
            removed_tetrahedron_count_before: 1,
            removed_tetrahedron_count_after: 4,
        })
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![10, 11, 12, 13]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("edge-star recovered cavity should remain valid");
}

#[test]
fn boundary_edge_star_recovery_queue_reports_ordered_steps() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2], [1, 2, 4, 5]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 20, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        &[[0, 1], [2, 4]],
        &[],
    )
    .expect("ordered boundary edge-star queue should evaluate");

    assert_eq!(
        recovery.steps,
        vec![
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [0, 1],
                added_tetrahedron_ids: vec![21, 22],
                removed_tetrahedron_count_before: 1,
                removed_tetrahedron_count_after: 3,
            },
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [2, 4],
                added_tetrahedron_ids: vec![23],
                removed_tetrahedron_count_before: 3,
                removed_tetrahedron_count_after: 4,
            }
        ]
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![20, 21, 22, 23]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("queued edge-star recovered cavity should remain valid");
}
