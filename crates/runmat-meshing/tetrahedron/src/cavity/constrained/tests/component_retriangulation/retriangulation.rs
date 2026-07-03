use super::super::*;

#[test]
fn component_retriangulation_from_nodes_preserves_boundary_and_volume() {
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

    let refill = retriangulate_constrained_cavity_from_nodes(&component_cavity, &nodes, options)
        .expect("component retriangulation should evaluate")
        .expect("component should have an exact cover");

    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("component retriangulation should preserve boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("component retriangulation should preserve volume");
}

#[test]
fn component_retriangulation_rejects_duplicate_node_ids() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 3,
        coordinates_m: [0.1, 0.1, 0.1],
    });

    let err = retriangulate_constrained_cavity_from_nodes(&cavity, &nodes, refill_options())
        .expect_err("duplicate node ids should be rejected");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::DuplicateInteriorNode { node_id: 3 }
    );
}
