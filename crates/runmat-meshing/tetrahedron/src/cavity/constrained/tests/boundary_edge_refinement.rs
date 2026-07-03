use super::*;

#[test]
fn boundary_edge_split_refines_conforming_faces_and_preserves_valid_cavity() {
    let cavity = provenance_cavity();
    let nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_node) =
        split_constrained_cavity_boundary_edge(&cavity, &nodes, [1, 0])
            .expect("boundary edge should split");

    assert_eq!(split_node.node_id, 4);
    assert_eq!(split_node.coordinates_m, [0.5, 0.0, 0.0]);
    assert_eq!(
        split_cavity.boundary_faces.len(),
        cavity.boundary_faces.len() + 2
    );
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert_eq!(
        split_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&split_node.node_id))
            .count(),
        4
    );
    validate_constrained_cavity(&split_cavity).expect("split cavity should remain valid");
}

#[test]
fn boundary_edge_patch_split_refines_pair_without_shared_edge_child() {
    let cavity = provenance_cavity();
    let nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_node) =
        split_constrained_cavity_boundary_edge_patch_at_centroid(&cavity, &nodes, [1, 0])
            .expect("boundary edge patch should split");

    assert_eq!(split_node.node_id, 4);
    assert_eq!(split_node.coordinates_m, [0.25, 0.25, 0.25]);
    assert_eq!(
        split_cavity.boundary_faces.len(),
        cavity.boundary_faces.len() + 2
    );
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, split_node.node_id]));
    assert_eq!(
        split_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&split_node.node_id))
            .count(),
        4
    );
    for expected in [[0, 2, 4], [1, 2, 4], [0, 3, 4], [1, 3, 4]] {
        assert!(split_cavity
            .boundary_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == expected));
    }
    validate_constrained_cavity(&split_cavity).expect("patch split cavity should remain valid");
}

#[test]
fn boundary_edge_patch_split_honors_weighted_point() {
    let cavity = provenance_cavity();
    let nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_node) = split_constrained_cavity_boundary_edge_patch_with_weights(
        &cavity,
        &nodes,
        [1, 0],
        [0.1, 0.2, 0.3, 0.4],
    )
    .expect("weighted boundary edge patch should split");

    assert_eq!(split_node.node_id, 4);
    assert_eq!(split_node.coordinates_m, [0.2, 0.3, 0.4]);
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, split_node.node_id]));
    validate_constrained_cavity(&split_cavity)
        .expect("weighted patch split cavity should remain valid");
}

#[test]
fn boundary_patch_split_reports_ordered_edge_and_face_steps() {
    let cavity = provenance_cavity();
    let nodes = unit_tetrahedron_nodes();

    let split = split_constrained_cavity_boundary_patch_at_centroids(
        &cavity,
        &nodes,
        &[[1, 0]],
        &[[1, 3, 2]],
    )
    .expect("boundary patch split should evaluate");

    assert_eq!(
        split.steps,
        vec![
            ConstrainedCavityBoundaryPatchSplitStep::EdgePatch {
                node_ids: [0, 1],
                split_node_id: 4,
            },
            ConstrainedCavityBoundaryPatchSplitStep::Face {
                node_ids: [1, 2, 3],
                split_node_id: 5,
            }
        ]
    );
    assert_eq!(
        split
            .split_nodes
            .iter()
            .map(|node| node.node_id)
            .collect::<Vec<_>>(),
        vec![4, 5]
    );
    assert!(!split
        .cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split
        .cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert!(!split
        .cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [1, 2, 3]));
    validate_constrained_cavity(&split.cavity)
        .expect("boundary patch split cavity should remain valid");
}
