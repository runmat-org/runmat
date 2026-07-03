use super::*;

#[test]
fn boundary_face_split_preserves_source_face_regions_and_perimeter_edges() {
    let face = face_with_provenance(
        [0, 1, 2],
        10,
        [Some(100), Some(101), Some(102)],
        &["fixed", "loaded"],
    );

    let children = split_constrained_cavity_boundary_face(&face, 9).expect("face should split");

    assert_eq!(children.len(), 3);
    assert_eq!(children[0].node_ids, [0, 1, 9]);
    assert_eq!(children[1].node_ids, [1, 2, 9]);
    assert_eq!(children[2].node_ids, [2, 0, 9]);
    for child in &children {
        assert_eq!(child.source_face_id, Some(10));
        assert_eq!(
            sorted_region_ids(&child.region_ids),
            vec!["fixed".to_string(), "loaded".to_string()]
        );
    }
    assert_eq!(children[0].source_edge_ids, [Some(100), None, None]);
    assert_eq!(children[1].source_edge_ids, [Some(101), None, None]);
    assert_eq!(children[2].source_edge_ids, [Some(102), None, None]);
}

#[test]
fn boundary_face_list_split_replaces_only_target_face() {
    let cavity = provenance_cavity();

    let split_faces = split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [2, 1, 0], 9)
        .expect("target face should split");

    assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9))
            .count(),
        3
    );
    for untouched in cavity.boundary_faces.iter().skip(1) {
        assert!(split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
    }
}

#[test]
fn boundary_face_edge_split_preserves_source_face_regions_and_split_edge_provenance() {
    let face = face_with_provenance(
        [0, 1, 2],
        10,
        [Some(100), Some(101), Some(102)],
        &["fixed", "loaded"],
    );

    let children = split_constrained_cavity_boundary_face_on_edge(&face, [0, 1], 9)
        .expect("face edge should split");

    assert_eq!(children[0].node_ids, [0, 9, 2]);
    assert_eq!(children[1].node_ids, [9, 1, 2]);
    assert_eq!(children[0].source_edge_ids, [Some(100), None, Some(102)]);
    assert_eq!(children[1].source_edge_ids, [Some(100), Some(101), None]);
    for child in &children {
        assert_eq!(child.source_face_id, Some(10));
        assert_eq!(
            sorted_region_ids(&child.region_ids),
            vec!["fixed".to_string(), "loaded".to_string()]
        );
    }
}

#[test]
fn boundary_face_edge_split_list_replaces_conforming_edge_pair() {
    let cavity = provenance_cavity();

    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        [2, 1, 0],
        [1, 0],
        9,
    )
    .expect("target face edge should split");

    assert_eq!(split_faces.len(), cavity.boundary_faces.len() + 2);
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9))
            .count(),
        4
    );
    for untouched in cavity.boundary_faces.iter().skip(2) {
        assert!(split_faces
            .iter()
            .any(|face| sorted_face(face.node_ids) == sorted_face(untouched.node_ids)));
    }
}

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

#[test]
fn boundary_face_three_edge_split_refines_target_and_conforming_neighbors() {
    let cavity = provenance_cavity();
    let split_faces = split_constrained_cavity_boundary_faces_on_three_edges(
        &cavity.boundary_faces,
        [2, 1, 0],
        BTreeMap::from([([0, 1], 9), ([1, 2], 10), ([0, 2], 11)]),
    )
    .expect("target face edges should split");

    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 2]));
    assert!(!split_faces
        .iter()
        .any(|face| sorted_face(face.node_ids) == [0, 1, 3]));
    assert_eq!(
        split_faces
            .iter()
            .filter(|face| face.node_ids.contains(&9)
                || face.node_ids.contains(&10)
                || face.node_ids.contains(&11))
            .count(),
        10
    );
    let target_children = split_faces
        .iter()
        .filter(|face| {
            [9, 10, 11]
                .into_iter()
                .any(|node_id| face.node_ids.contains(&node_id))
                && face.source_face_id == Some(10)
        })
        .collect::<Vec<_>>();
    assert_eq!(target_children.len(), 4);
    assert_eq!(source_edge_for(target_children[0], [0, 9]), Some(100));
    assert_eq!(source_edge_for(target_children[1], [1, 9]), Some(100));
    assert_eq!(source_edge_for(target_children[1], [1, 10]), Some(101));
    assert_eq!(source_edge_for(target_children[2], [2, 10]), Some(101));
    assert_eq!(source_edge_for(target_children[2], [2, 11]), Some(102));
    assert_eq!(source_edge_for(target_children[0], [0, 11]), Some(102));
}

#[test]
fn boundary_face_split_rejects_reused_or_missing_split_targets() {
    let cavity = provenance_cavity();
    let face = &cavity.boundary_faces[0];

    let reused = split_constrained_cavity_boundary_face(face, face.node_ids[0])
        .expect_err("split node cannot reuse an existing face node");
    assert_eq!(
        reused,
        ConstrainedCavityBoundarySplitError::SplitNodeReusesFaceNode {
            node_id: face.node_ids[0]
        }
    );

    let missing = split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [10, 11, 12], 9)
        .expect_err("missing target face should fail");
    assert_eq!(
        missing,
        ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
            node_ids: [10, 11, 12]
        }
    );
}
