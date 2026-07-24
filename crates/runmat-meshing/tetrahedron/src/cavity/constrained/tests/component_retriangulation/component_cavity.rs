use super::super::*;

#[test]
fn refill_tetrahedron_component_cavity_preserves_boundary_metadata_and_volume() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let lower = raw_refill_tetrahedron_with_rejection_reason(
        [0, 1, 2, 3],
        [0, 1, 2, 3].map(|node_id| node_map[&node_id]),
        options,
    )
    .expect("lower bipyramid tetrahedron should pass quality gates");
    let upper = raw_refill_tetrahedron_with_rejection_reason(
        [0, 2, 1, 4],
        [0, 2, 1, 4].map(|node_id| node_map[&node_id]),
        options,
    )
    .expect("upper bipyramid tetrahedron should pass quality gates");

    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &[lower, upper],
        &source_cavity.boundary_faces,
        vec![0],
    )
    .expect("selected component should define a valid cavity");

    assert_eq!(component_cavity.removed_tetrahedron_ids, vec![0, 1]);
    assert_eq!(component_cavity.boundary_faces.len(), 6);
    assert_eq!(component_cavity.protected_node_ids, vec![0]);
    assert!((component_cavity.target_volume_m3 - source_cavity.target_volume_m3).abs() < 1.0e-12);
    assert!(!component_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    let inherited_face = component_cavity
        .boundary_faces
        .iter()
        .find(|face| sorted_face(face.node_ids) == [0, 1, 3])
        .expect("component cavity should preserve inherited source face");
    assert_eq!(inherited_face.region_ids, vec!["body".to_string()]);
}

#[test]
fn refill_tetrahedron_component_cavity_round_trips_through_refill_evaluation() {
    let source_cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map = boundary_node_coordinates(&source_cavity, &nodes)
        .expect("fixture nodes should cover source boundary");
    let options = refill_options();
    let component_tetrahedra = [([0, 1, 2, 3], [0, 1, 2, 3]), ([0, 2, 1, 4], [0, 2, 1, 4])]
        .into_iter()
        .map(|(node_ids, point_ids)| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                point_ids.map(|node_id| node_map[&node_id]),
                options,
            )
            .expect("component tetrahedron should pass quality gates")
        })
        .collect::<Vec<_>>();
    let component_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &component_tetrahedra,
        &source_cavity.boundary_faces,
        Vec::new(),
    )
    .expect("selected component should define a valid cavity");

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&component_cavity, &nodes, &[], options)
            .expect("component cavity refill should evaluate");
    let refill = evaluation
        .refill
        .expect("component cavity should be refillable");

    assert_eq!(refill.tetrahedra.len(), component_tetrahedra.len());
    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("component refill should preserve derived boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("component refill should preserve derived volume");
}
