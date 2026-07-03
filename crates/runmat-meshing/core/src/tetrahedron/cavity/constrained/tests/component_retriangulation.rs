use super::*;

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
