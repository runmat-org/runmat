use super::super::*;

#[test]
fn boundary_face_three_edge_split_completion_reports_inserted_nodes_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_nodes, split_tetrahedra) =
        best_boundary_face_three_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("three-edge completion should evaluate")
        .expect("three-edge completion should generate child cap tetrahedra");

    assert_eq!(inserted_nodes.len(), 3);
    assert_eq!(
        inserted_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert!(inserted_nodes
        .iter()
        .all(|node| node.coordinates_m[2].abs() <= 1.0e-12));
    assert_eq!(split_tetrahedra.len(), 4);
    assert!(split_tetrahedra.iter().all(|tetrahedron| {
        inserted_nodes
            .iter()
            .any(|node| tetrahedron.node_ids.contains(&node.node_id))
    }));
    assert!(!refined_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        refined_cavity
            .boundary_faces
            .iter()
            .filter(|face| inserted_nodes
                .iter()
                .any(|node| face.node_ids.contains(&node.node_id)))
            .count(),
        10
    );

    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("three-edge child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("three-edge completion should preserve the original target volume");
}
