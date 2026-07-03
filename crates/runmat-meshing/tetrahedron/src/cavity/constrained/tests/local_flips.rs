use super::*;

#[test]
fn shared_face_flip_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let flipped_tetrahedra =
        flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 2], options)
            .expect("shared face should flip");

    assert_eq!(flipped_tetrahedra.len(), 3);
    assert!(flipped_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&3) && tetrahedron.node_ids.contains(&4)));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("face flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("face flip should preserve target volume");
}

#[test]
fn shared_face_flip_rejects_boundary_face() {
    let nodes = two_tetrahedron_face_flip_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_across_shared_face(&tetrahedra, &nodes, [0, 1, 3], options)
        .expect_err("boundary face should not have two incident tetrahedra");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::FaceIncidenceNotTwo {
            node_ids: [0, 1, 3],
            incident_tetrahedron_count: 1,
        }
    );
}

#[test]
fn shared_edge_flip_preserves_component_boundary_and_volume() {
    let nodes = triangular_edge_ring_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 3, 4, 5], [0, 4, 3, 6], [0, 5, 6, 3]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| {
                    nodes
                        .iter()
                        .find(|node| node.node_id == node_id)
                        .expect("fixture node should exist")
                        .coordinates_m
                }),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_refill_tetrahedron_component(&tetrahedra, &[], Vec::new())
        .expect("edge-ring component should define a valid cavity");

    let flipped_tetrahedra =
        flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 3], options)
            .expect("three-tetrahedron edge ring should flip");

    assert_eq!(flipped_tetrahedra.len(), 2);
    assert!(flipped_tetrahedra.iter().all(|tetrahedron| [4, 5, 6]
        .iter()
        .all(|node_id| tetrahedron.node_ids.contains(node_id))));
    let flipped_cavity = constrained_cavity_from_refill_tetrahedron_component(
        &flipped_tetrahedra,
        &cavity.boundary_faces,
        Vec::new(),
    )
    .expect("flipped edge-ring component should remain a valid cavity");
    validate_constrained_cavity_boundary_preserved(&cavity, &flipped_cavity.boundary_faces)
        .expect("edge flip should preserve the component boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        flipped_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
        options.volume_relative_tolerance,
    )
    .expect("edge flip should preserve target volume");
}

#[test]
fn shared_edge_flip_rejects_non_three_tetrahedron_ring() {
    let nodes = two_tetrahedron_bipyramid_nodes();
    let options = refill_options();
    let tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| nodes[node_id as usize].coordinates_m),
                options,
            )
            .expect("fixture tetrahedron should pass quality")
        })
        .collect::<Vec<_>>();

    let err = flip_refill_tetrahedra_around_shared_edge(&tetrahedra, &nodes, [0, 1], options)
        .expect_err("two-tetrahedron edge should not be a three-tetrahedron flip ring");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronFlipError::EdgeIncidenceNotThree {
            node_ids: [0, 1],
            incident_tetrahedron_count: 2,
        }
    );
}
