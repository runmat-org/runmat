use super::super::*;
use runmat_meshing_core::NeverCancelled;

#[test]
fn component_steiner_nodes_are_bounded_inside_and_retriangulatable() {
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

    let steiner_nodes = generate_constrained_cavity_component_steiner_nodes(
        &component_cavity,
        &nodes,
        &component_tetrahedra,
        options,
        4,
    )
    .expect("component Steiner generation should evaluate");

    assert_eq!(steiner_nodes.len(), 4);
    assert_eq!(
        steiner_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![5, 6, 7, 8]
    );
    let boundary_node_map = boundary_node_coordinates(&component_cavity, &nodes)
        .expect("fixture nodes should cover component boundary");
    let boundary_triangles = cavity_boundary_triangles(&component_cavity, &boundary_node_map)
        .expect("component boundary should build triangles");
    assert!(steiner_nodes.iter().all(|node| {
        point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
    }));
    let mut nodes_with_steiner = nodes.clone();
    nodes_with_steiner.extend(steiner_nodes);
    let refill = retriangulate_constrained_cavity_from_nodes(
        &component_cavity,
        &nodes_with_steiner,
        options,
        ConstrainedCavityRefillBudget::default(),
        &NeverCancelled,
    )
    .expect("Steiner component retriangulation should evaluate")
    .expect("component should remain retriangulatable with generated Steiner nodes");
    validate_constrained_cavity_boundary_preserved(&component_cavity, &refill.boundary_faces)
        .expect("Steiner retriangulation should preserve boundary");
    validate_constrained_cavity_refill_volume(
        component_cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("Steiner retriangulation should preserve volume");
}

#[test]
fn patch_steiner_nodes_are_empty_for_boundary_complete_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let steiner_nodes =
        generate_constrained_cavity_patch_steiner_nodes(&cavity, &nodes, refill_options(), 4)
            .expect("patch Steiner generation should evaluate");

    assert!(steiner_nodes.is_empty());
}

#[test]
fn patch_steiner_nodes_are_bounded_inside_and_unique() {
    let cavity = unit_cube_cavity();
    let nodes = unit_cube_nodes();
    let options = refill_options();

    let steiner_nodes =
        generate_constrained_cavity_patch_steiner_nodes(&cavity, &nodes, options, 4)
            .expect("patch Steiner generation should evaluate");

    assert_eq!(steiner_nodes.len(), 4);
    assert_eq!(
        steiner_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![8, 9, 10, 11]
    );
    let boundary_node_map = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_node_map)
        .expect("cavity boundary should build triangles");
    assert!(steiner_nodes.iter().all(|node| {
        point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
    }));
    let mut all_node_ids = nodes
        .iter()
        .map(|node| node.node_id)
        .collect::<BTreeSet<_>>();
    for node in &steiner_nodes {
        assert!(all_node_ids.insert(node.node_id));
    }
}
