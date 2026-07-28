use super::super::*;

#[test]
fn boundary_face_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert!(inserted_node.coordinates_m[0] > 0.0);
    assert!(inserted_node.coordinates_m[1] > 0.0);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] < 1.0);
    assert_eq!(split_tetrahedra.len(), 3);
    assert!(split_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&inserted_node.node_id)));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&inserted_node.node_id))
            .count(),
        3
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("split completion should preserve the original target volume");
}

#[test]
fn boundary_face_split_completion_prefers_higher_quality_split_point() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: tetrahedron_faces([0, 1, 2, 3])
            .into_iter()
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: vec!["body".to_string()],
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 2.0 / 3.0,
    };
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.649331064611886, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.10383330216927095, 0.5285988568010986, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.583996624105325, 0.04591313203731445, 1.25490017426856],
        },
    ];
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let centroid_node = boundary_face_centroid_node([0, 1, 2], &boundary_nodes);
    let centroid_tetrahedra = split_completion_tetrahedra_for_node(
        [0, 1, 2],
        3,
        &centroid_node,
        &boundary_nodes,
        options,
    )
    .expect("centroid split should generate child cap tetrahedra");
    let centroid_min_quality = centroid_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    let (_, inserted_node, split_tetrahedra) = best_boundary_face_split_completion(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &[],
        options,
    )
    .expect("split completion should evaluate")
    .expect("split completion should generate child cap tetrahedra");
    let selected_min_quality = split_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);

    assert!(
            selected_min_quality > centroid_min_quality + 1.0e-9,
            "split search should improve on the centroid split: selected={selected_min_quality} centroid={centroid_min_quality}"
        );
    assert_ne!(inserted_node.coordinates_m, centroid_node.coordinates_m);
}
