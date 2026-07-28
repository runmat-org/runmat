use super::super::*;

#[test]
fn boundary_face_centroid_split_refines_target_face_and_remains_refillable() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_node) =
        split_constrained_cavity_boundary_face_at_centroid(&cavity, &nodes, [2, 1, 0])
            .expect("boundary face should split at centroid");
    nodes.push(split_node.clone());

    assert_eq!(split_node.node_id, 4);
    assert_eq!(split_node.coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert_eq!(
        split_cavity.boundary_faces.len(),
        cavity.boundary_faces.len() + 2
    );
    assert!(!split_cavity
        .boundary_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        split_cavity
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&split_node.node_id))
            .count(),
        3
    );
    validate_constrained_cavity(&split_cavity).expect("split cavity should remain valid");

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&split_cavity, &nodes, &[], refill_options())
            .expect("split cavity refill should evaluate");
    let refill = evaluation
        .refill
        .expect("split boundary cavity should remain refillable");
    validate_constrained_cavity_boundary_preserved(&split_cavity, &refill.boundary_faces)
        .expect("split refill should preserve refined boundary");
    validate_constrained_cavity_refill_volume(
        split_cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("split refill should preserve volume");
}

#[test]
fn boundary_face_barycentric_split_places_node_and_remains_refillable() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_node) = split_constrained_cavity_boundary_face_at_barycentric(
        &cavity,
        &nodes,
        [2, 1, 0],
        [0.5, 0.25, 0.25],
    )
    .expect("boundary face should split at requested barycentric point");
    nodes.push(split_node.clone());

    assert_eq!(split_node.node_id, 4);
    assert_eq!(split_node.coordinates_m, [0.25, 0.25, 0.0]);
    validate_constrained_cavity(&split_cavity).expect("split cavity should remain valid");
    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&split_cavity, &nodes, &[], refill_options())
            .expect("split cavity refill should evaluate");
    let refill = evaluation
        .refill
        .expect("split boundary cavity should remain refillable");
    validate_constrained_cavity_boundary_preserved(&split_cavity, &refill.boundary_faces)
        .expect("split refill should preserve refined boundary");
    validate_constrained_cavity_refill_volume(
        split_cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("split refill should preserve volume");
}

#[test]
fn boundary_face_barycentric_split_rejects_invalid_coordinates() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();

    let err = split_constrained_cavity_boundary_face_at_barycentric(
        &cavity,
        &nodes,
        [0, 1, 2],
        [0.5, 0.5, 0.5],
    )
    .expect_err("barycentric coordinates should sum to one");

    assert_eq!(
        err,
        ConstrainedCavityBoundaryFaceSplitError::InvalidBarycentricCoordinates {
            barycentric: [0.5, 0.5, 0.5]
        }
    );
}

#[test]
fn boundary_face_centroid_split_refines_adjacent_face_pair_and_remains_refillable() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();

    let (split_cavity, split_nodes) = split_constrained_cavity_boundary_faces_at_centroids(
        &cavity,
        &nodes,
        &[[2, 1, 0], [3, 1, 0]],
    )
    .expect("adjacent boundary faces should split at centroids");
    nodes.extend(split_nodes.clone());

    assert_eq!(split_nodes.len(), 2);
    assert_eq!(split_nodes[0].node_id, 4);
    assert_eq!(split_nodes[1].node_id, 5);
    assert_eq!(split_nodes[0].coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert_eq!(split_nodes[1].coordinates_m, [1.0 / 3.0, 0.0, 1.0 / 3.0]);
    assert_eq!(
        split_cavity.boundary_faces.len(),
        cavity.boundary_faces.len() + 4
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
            .filter(|face| face.node_ids.contains(&4) || face.node_ids.contains(&5))
            .count(),
        6
    );
    validate_constrained_cavity(&split_cavity).expect("split cavity should remain valid");

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&split_cavity, &nodes, &[], refill_options())
            .expect("split cavity refill should evaluate");
    let refill = evaluation
        .refill
        .expect("split boundary cavity should remain refillable");
    validate_constrained_cavity_boundary_preserved(&split_cavity, &refill.boundary_faces)
        .expect("split refill should preserve refined boundary");
    validate_constrained_cavity_refill_volume(
        split_cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("split refill should preserve volume");
}

#[test]
fn boundary_face_centroid_split_rejects_duplicate_target_face() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();

    let err = split_constrained_cavity_boundary_faces_at_centroids(
        &cavity,
        &nodes,
        &[[0, 1, 2], [2, 1, 0]],
    )
    .expect_err("duplicate split targets should be rejected");

    assert_eq!(
        err,
        ConstrainedCavityBoundaryFaceSplitError::DuplicateBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}
