use super::*;

#[test]
fn extracts_single_tetrahedron_cavity_from_selected_tetrahedra() {
    let tetrahedra = vec![candidate_tetrahedron(7, [0, 1, 2, 3], 0.25, &["body"])];

    let cavity = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[0], vec![0, 1])
        .expect("single tetrahedron cavity should extract");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![7]);
    assert_eq!(cavity.boundary_faces.len(), 4);
    assert_eq!(cavity.protected_node_ids, vec![0, 1]);
    assert_eq!(cavity.target_volume_m3, 0.25);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids == ["body"]));
}

#[test]
fn extracts_boundary_faces_from_two_tetrahedron_cavity() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["left"]),
        candidate_tetrahedron(9, [0, 2, 1, 4], 0.35, &["right"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[1, 0], vec![])
        .expect("two tetrahedron cavity should extract");

    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();

    assert_eq!(cavity.removed_tetrahedron_ids, vec![4, 9]);
    assert_eq!(cavity.boundary_faces.len(), 6);
    assert!(!boundary_faces.contains(&[0, 1, 2]));
    assert_eq!(cavity.target_volume_m3, 0.60);
    validate_constrained_cavity(&cavity).expect("extracted cavity should validate");
}

#[test]
fn extracted_cavity_tracks_outside_neighbor_tetrahedra() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["left"]),
        candidate_tetrahedron(9, [0, 2, 1, 4], 0.35, &["right"]),
        candidate_tetrahedron(12, [0, 4, 2, 5], 0.20, &["outside"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[0, 1], vec![])
        .expect("two selected tetrahedra should extract");

    let outside_face = cavity
        .boundary_faces
        .iter()
        .find(|face| sorted_face(face.node_ids) == [0, 2, 4])
        .expect("shared face with untouched neighbor should remain on cavity boundary");
    assert_eq!(outside_face.outside_tetrahedron_ids, vec![12]);
    assert!(cavity
        .boundary_faces
        .iter()
        .filter(|face| sorted_face(face.node_ids) != [0, 2, 4])
        .all(|face| face.outside_tetrahedron_ids.is_empty()));
    validate_constrained_cavity(&cavity).expect("extracted cavity should validate");
}

#[test]
fn rejects_duplicate_selected_tetrahedron_indices() {
    let tetrahedra = vec![candidate_tetrahedron(7, [0, 1, 2, 3], 0.25, &[])];

    let err = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[0, 0], vec![])
        .expect_err("duplicate selection should fail");

    assert_eq!(
        err,
        ConstrainedCavityExtractionError::DuplicateSelectedTetrahedronIndex {
            tetrahedron_index: 0
        }
    );
}

#[test]
fn rejects_selected_tetrahedron_indices_out_of_bounds() {
    let tetrahedra = vec![candidate_tetrahedron(7, [0, 1, 2, 3], 0.25, &[])];

    let err = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[1], vec![])
        .expect_err("out of bounds selection should fail");

    assert_eq!(
        err,
        ConstrainedCavityExtractionError::SelectedTetrahedronIndexOutOfBounds {
            tetrahedron_index: 1,
            tetrahedron_count: 1
        }
    );
}

#[test]
fn rejects_selected_tetrahedra_with_open_boundary() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &[]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &[]),
    ];

    let err = constrained_cavity_from_selected_tetrahedra(&tetrahedra, &[0, 1], vec![])
        .expect_err("nonmanifold selected cavity should fail");

    assert_eq!(
        err,
        ConstrainedCavityExtractionError::Validation(
            ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                node_ids: [0, 1],
                face_count: 4
            }
        )
    );
}

#[test]
fn anchor_trim_removes_non_manifold_dangling_tetrahedron() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["anchor"]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &["dangling"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1],
        0,
        vec![0, 1],
    )
    .expect("trim should evaluate")
    .expect("trim should recover the anchor tetrahedron cavity");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![4]);
    assert_eq!(cavity.target_volume_m3, 0.25);
    assert_eq!(cavity.protected_node_ids, vec![0, 1]);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids == ["anchor"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_preserves_requested_anchor() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["left"]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &["right"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1],
        1,
        vec![],
    )
    .expect("trim should evaluate")
    .expect("trim should keep the requested anchor tetrahedron");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![9]);
    assert_eq!(cavity.target_volume_m3, 0.35);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids == ["right"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_searches_past_first_defective_edge() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &["anchor"]),
        candidate_tetrahedron(9, [0, 1, 2, 4], 0.35, &["trimmed"]),
        candidate_tetrahedron(11, [0, 1, 2, 5], 0.45, &["kept"]),
        candidate_tetrahedron(13, [0, 1, 4, 5], 0.55, &["kept"]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0, 1, 2, 3],
        0,
        vec![],
    )
    .expect("trim should evaluate")
    .expect("trim should find an anchor-containing manifold subset");

    assert_eq!(cavity.removed_tetrahedron_ids, vec![4, 11, 13]);
    assert_eq!(cavity.target_volume_m3, 1.25);
    assert!(cavity
        .boundary_faces
        .iter()
        .all(|face| face.region_ids != ["trimmed"]));
    validate_constrained_cavity(&cavity).expect("trimmed cavity should be manifold");
}

#[test]
fn anchor_trim_returns_none_when_anchor_not_selected() {
    let tetrahedra = vec![
        candidate_tetrahedron(4, [0, 1, 2, 3], 0.25, &[]),
        candidate_tetrahedron(9, [0, 1, 4, 5], 0.35, &[]),
    ];

    let cavity = constrained_cavity_from_selected_tetrahedra_with_anchor_trim(
        &tetrahedra,
        &[0],
        1,
        Vec::new(),
    )
    .expect("trim should evaluate");

    assert!(cavity.is_none());
}

#[test]
fn boundary_preservation_accepts_reoriented_faces_with_same_provenance() {
    let cavity = provenance_cavity();
    let candidate_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            let mut reoriented = face.clone();
            reoriented.node_ids = [face.node_ids[2], face.node_ids[1], face.node_ids[0]];
            reoriented.source_edge_ids = [
                source_edge_for(face, [reoriented.node_ids[0], reoriented.node_ids[1]]),
                source_edge_for(face, [reoriented.node_ids[1], reoriented.node_ids[2]]),
                source_edge_for(face, [reoriented.node_ids[2], reoriented.node_ids[0]]),
            ];
            reoriented.region_ids.reverse();
            reoriented
        })
        .collect::<Vec<_>>();

    validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect("same boundary and provenance should validate");
}

#[test]
fn boundary_preservation_rejects_missing_boundary_face() {
    let cavity = provenance_cavity();
    let mut candidate_faces = cavity.boundary_faces.clone();
    candidate_faces[0].node_ids = [10, 11, 12];

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("missing boundary face should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::MissingBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}

#[test]
fn boundary_preservation_rejects_source_face_mismatch() {
    let cavity = provenance_cavity();
    let mut candidate_faces = cavity.boundary_faces.clone();
    candidate_faces[0].source_face_id = Some(99);

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("source face mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundarySourceFaceMismatch {
            node_ids: [0, 1, 2],
            expected_source_face_id: Some(10),
            candidate_source_face_id: Some(99)
        }
    );
}

#[test]
fn boundary_preservation_rejects_source_edge_mismatch() {
    let cavity = provenance_cavity();
    let mut candidate_faces = cavity.boundary_faces.clone();
    candidate_faces[0].source_edge_ids[0] = Some(99);

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("source edge mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundarySourceEdgeMismatch {
            node_ids: [0, 1],
            expected_source_edge_id: Some(100),
            candidate_source_edge_id: Some(99)
        }
    );
}

#[test]
fn boundary_preservation_rejects_region_mismatch() {
    let cavity = provenance_cavity();
    let mut candidate_faces = cavity.boundary_faces.clone();
    candidate_faces[0].region_ids = vec!["other".to_string()];

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("region mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundaryRegionMismatch {
            node_ids: [0, 1, 2],
            expected_region_ids: vec!["fixed".to_string(), "loaded".to_string()],
            candidate_region_ids: vec!["other".to_string()]
        }
    );
}

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

#[test]
fn refill_candidates_preserve_split_boundary_face() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.boundary_faces =
        split_constrained_cavity_boundary_faces(&cavity.boundary_faces, [0, 1, 2], 4)
            .expect("fixture face should split");
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 4,
        coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("split boundary cavity should refill");

    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill should preserve split boundary faces");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("split boundary refill should preserve volume");
    assert!(
        refill
            .boundary_faces
            .iter()
            .filter(|face| face.node_ids.contains(&4))
            .count()
            >= 3
    );
}

#[test]
fn refill_candidates_preserve_single_tetrahedron_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("single tetrahedron cavity should refill");

    assert_eq!(refill.tetrahedra.len(), 1);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("refill boundary should match cavity boundary");
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn single_tetrahedron_refill_ignores_non_boundary_nodes_in_coordinate_table() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.push(ConstrainedCavityNode {
        node_id: 99,
        coordinates_m: [4.0, 4.0, 4.0],
    });

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("coordinate table may contain nodes outside the cavity boundary");

    assert_eq!(refill.tetrahedra.len(), 1);
    assert!((refill.total_volume_m3 - cavity.target_volume_m3).abs() < 1.0e-12);
}

#[test]
fn star_refill_candidates_preserve_cavity_boundary_and_volume() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let interior = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let refill =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &interior, refill_options())
            .expect("interior star refill should generate");

    assert_eq!(refill.tetrahedra.len(), 4);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("star refill boundary should match cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("star refill should preserve cavity volume");
}

#[test]
fn refill_candidates_reject_missing_boundary_nodes() {
    let cavity = unit_tetrahedron_cavity();
    let mut nodes = unit_tetrahedron_nodes();
    nodes.pop();

    let err = generate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
        .expect_err("missing boundary node should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::MissingBoundaryNode { node_id: 3 }
    );
}

#[test]
fn star_refill_candidates_reject_exterior_interior_points() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let exterior = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [2.0, 2.0, 2.0],
    }];

    let err =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &exterior, refill_options())
            .expect_err("exterior interior candidate should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
        }
    );
}

#[test]
fn star_refill_evaluation_reports_scaled_jacobian_rejections() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let near_corner = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [1.0e-4, 1.0e-4, 1.0e-4],
    }];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &near_corner,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.5,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("evaluation should classify a low-quality star candidate");

    assert!(evaluation.refill.is_none());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("star_tetrahedron_scaled_jacobian".to_string(), 1)])
    );
}

#[test]
fn boundary_node_refill_evaluation_reports_contextual_scaled_jacobian_rejections() {
    let cavity = octahedron_cavity();
    let nodes = octahedron_nodes();

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &[],
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("boundary-node evaluation should classify low-quality candidates");

    assert!(evaluation.refill.is_none());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([
            ("boundary_node_tetrahedron_scaled_jacobian".to_string(), 1),
            (
                "centroid_interior_refill_tetrahedron_scaled_jacobian".to_string(),
                1,
            ),
        ])
    );
}

#[test]
fn refill_evaluation_uses_boundary_nodes_for_multi_face_cavity_without_interior_point() {
    let cavity = octahedron_cavity();
    let nodes = octahedron_nodes();

    let evaluation =
        evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], refill_options())
            .expect("evaluation should complete");

    let refill = evaluation
        .refill
        .expect("boundary-node refill should support closed multi-face cavities");
    assert!(evaluation.rejected_by_reason.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("boundary-node refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        1.0e-12,
    )
    .expect("boundary-node refill should preserve volume");
}

#[test]
fn boundary_node_completion_repairs_missing_cavity_boundary_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert!(refill_from_tetrahedra(
        &cavity,
        vec![incomplete_tetrahedron.clone()],
        options.volume_relative_tolerance
    )
    .is_err());

    let (_, completed, inserted_nodes) = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        options,
    )
    .expect("completion should evaluate")
    .expect("completion should add the missing tetrahedron");
    let refill = refill_from_tetrahedra(&cavity, completed, options.volume_relative_tolerance)
        .expect("completed refill should validate");

    assert!(inserted_nodes.is_empty());
    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("completed refill should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("completed refill should preserve volume");
}

#[test]
fn boundary_node_completion_reports_when_no_cap_tetrahedron_passes_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let initial_options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let incomplete_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, initial_options)
            .expect("fixture tetrahedron should pass initial quality gates");

    let rejected = complete_missing_boundary_face_tetrahedra(
        &cavity,
        &boundary_nodes,
        vec![incomplete_tetrahedron],
        &boundary_triangles,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            ..initial_options
        },
    )
    .expect("completion should evaluate")
    .expect_err("strict quality should reject every cap tetrahedron");

    assert_eq!(rejected, "boundary_node_completion_no_candidate");
}

#[test]
fn boundary_node_exact_cover_recovers_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("exact cover should recover the cavity");

    assert_eq!(refill.tetrahedra.len(), 2);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve volume");
}

#[test]
fn boundary_cap_nodes_are_empty_when_solid_boundary_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let cap_nodes =
        generate_constrained_cavity_boundary_cap_nodes(&cavity, &nodes, refill_options(), 4)
            .expect("boundary cap node generation should evaluate");

    assert!(cap_nodes.is_empty());
}

#[test]
fn solid_empty_boundary_faces_are_empty_when_boundary_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let faces = constrained_cavity_solid_empty_boundary_faces(&cavity, &nodes, refill_options())
        .expect("solid-empty boundary face detection should evaluate");

    assert!(faces.is_empty());
}

#[test]
fn solid_empty_boundary_face_classification_is_empty_when_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let classification =
        constrained_cavity_classified_solid_empty_boundary_faces(&cavity, &nodes, refill_options())
            .expect("solid-empty boundary face classification should evaluate");

    assert_eq!(
        classification,
        ConstrainedCavitySolidEmptyBoundaryFaces {
            faces: Vec::new(),
            true_exterior_faces: Vec::new(),
            expandable_faces: Vec::new(),
        }
    );
}

#[test]
fn solid_empty_boundary_recovery_is_noop_when_coverage_exists() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let source_tetrahedra = [
        candidate_tetrahedron(1, [0, 1, 2, 3], 1.0 / 6.0, &["body"]),
        candidate_tetrahedron(2, [0, 2, 1, 4], 1.0 / 6.0, &["body"]),
    ];

    let recovery = recover_constrained_cavity_solid_empty_boundaries(
        &cavity,
        &nodes,
        &source_tetrahedra,
        &nodes,
        refill_options(),
    )
    .expect("solid-empty boundary recovery should evaluate");

    assert_eq!(recovery.cavity, cavity);
    assert!(recovery.split_nodes.is_empty());
    assert!(recovery.split_steps.is_empty());
    assert!(recovery.rejected_splits.is_empty());
    assert!(recovery.expanded_removed_tetrahedron_ids.is_empty());
    assert!(recovery.classification.faces.is_empty());
}

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
fn boundary_edge_star_recovery_reports_added_tetrahedra() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 10, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        [1, 0],
        &[],
    )
    .expect("boundary edge-star recovery should evaluate");

    assert_eq!(recovery.attempted_boundary_faces, Vec::<[u32; 3]>::new());
    assert_eq!(
        recovery.recovered_edge,
        Some(ConstrainedCavityBoundaryEdgeRecoveryStep {
            node_ids: [0, 1],
            added_tetrahedron_ids: vec![11, 12, 13],
            removed_tetrahedron_count_before: 1,
            removed_tetrahedron_count_after: 4,
        })
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![10, 11, 12, 13]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("edge-star recovered cavity should remain valid");
}

#[test]
fn boundary_edge_star_recovery_queue_reports_ordered_steps() {
    let source_tetrahedra = [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2], [1, 2, 4, 5]]
        .into_iter()
        .enumerate()
        .map(|(index, node_ids)| candidate_tetrahedron(index as u32 + 20, node_ids, 1.0, &["body"]))
        .collect::<Vec<_>>();
    let cavity = constrained_cavity_from_selected_tetrahedra(&source_tetrahedra, &[0], Vec::new())
        .expect("single-tetrahedron source cavity should extract");

    let recovery = constrained_cavity_recovered_boundary_edge_star_queue_excluding_nodes(
        &cavity,
        &source_tetrahedra,
        &[[0, 1], [2, 4]],
        &[],
    )
    .expect("ordered boundary edge-star queue should evaluate");

    assert_eq!(
        recovery.steps,
        vec![
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [0, 1],
                added_tetrahedron_ids: vec![21, 22],
                removed_tetrahedron_count_before: 1,
                removed_tetrahedron_count_after: 3,
            },
            ConstrainedCavityBoundaryEdgeRecoveryStep {
                node_ids: [2, 4],
                added_tetrahedron_ids: vec![23],
                removed_tetrahedron_count_before: 3,
                removed_tetrahedron_count_after: 4,
            }
        ]
    );
    assert_eq!(
        recovery.cavity.removed_tetrahedron_ids,
        vec![20, 21, 22, 23]
    );
    validate_constrained_cavity(&recovery.cavity)
        .expect("queued edge-star recovered cavity should remain valid");
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

#[test]
fn shared_face_split_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
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

    let (split_tetrahedra, split_node) = split_refill_tetrahedra_across_shared_face_at_barycentric(
        &component_tetrahedra,
        &nodes,
        [0, 1, 2],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        options,
    )
    .expect("shared face should split");
    let refill =
        refill_from_tetrahedra(&cavity, split_tetrahedra, options.volume_relative_tolerance)
            .expect("shared-face split should preserve cavity boundary");

    assert_eq!(split_node.node_id, 5);
    assert_eq!(split_node.coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert_eq!(refill.tetrahedra.len(), 6);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("shared-face split should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("shared-face split should preserve volume");
}

#[test]
fn shared_face_split_composes_and_preserves_component_boundary_and_volume() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
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

    let (first_split_tetrahedra, first_split_node) =
        split_refill_tetrahedra_across_shared_face_at_barycentric(
            &component_tetrahedra,
            &nodes,
            [0, 1, 2],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            options,
        )
        .expect("first shared face should split");
    nodes.push(first_split_node.clone());
    let (second_split_tetrahedra, second_split_node) =
        split_refill_tetrahedra_across_shared_face_at_barycentric(
            &first_split_tetrahedra,
            &nodes,
            [0, 1, first_split_node.node_id],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            options,
        )
        .expect("new shared child face should split");
    let refill = refill_from_tetrahedra(
        &cavity,
        second_split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("composed shared-face split should preserve cavity boundary");

    assert_eq!(first_split_node.node_id, 5);
    assert_eq!(second_split_node.node_id, 6);
    assert_eq!(refill.tetrahedra.len(), 10);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("composed shared-face split should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("composed shared-face split should preserve volume");
}

#[test]
fn shared_face_split_rejects_non_shared_face() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let node_map =
        boundary_node_coordinates(&cavity, &nodes).expect("fixture nodes should cover cavity");
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

    let err = split_refill_tetrahedra_across_shared_face_at_barycentric(
        &component_tetrahedra,
        &nodes,
        [0, 1, 3],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        options,
    )
    .expect_err("boundary face should not split as a shared interior face");

    assert_eq!(
        err,
        ConstrainedCavityRefillTetrahedronSplitError::FaceIncidenceNotTwo {
            node_ids: [0, 1, 3],
            incident_tetrahedron_count: 1
        }
    );
}

#[test]
fn boundary_node_refill_applies_quality_gated_two_to_three_flip() {
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.05, 0.55, 0.3],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.55, 0.05, -0.3],
        },
    ];
    let boundary_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let options = refill_options();
    let baseline_tetrahedra = [[0, 1, 2, 3], [0, 2, 1, 4]]
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| boundary_nodes[&node_id]),
                options,
            )
            .expect("baseline tetrahedron should pass fixture quality gates")
        })
        .collect::<Vec<_>>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
        ]
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
        target_volume_m3: baseline_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
    };
    let baseline = refill_from_tetrahedra(
        &cavity,
        baseline_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("baseline should preserve the cavity boundary");

    let evaluation = evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], options)
        .expect("refill evaluation should complete");

    let refill = evaluation.refill.expect("boundary-node refill should pass");
    assert_eq!(refill.tetrahedra.len(), 3);
    assert!(refill_is_better(&refill, &baseline));
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 1, 3, 4]),
            sorted_tetrahedron_nodes([1, 2, 3, 4]),
            sorted_tetrahedron_nodes([0, 2, 3, 4])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("flipped refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("flipped refill should preserve volume");
}

#[test]
fn boundary_node_refill_applies_quality_gated_three_to_two_flip() {
    let nodes = vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.45, 0.5, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.5, 0.45, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [0.0, 1.0, 0.0],
        },
    ];
    let boundary_nodes = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let options = refill_options();
    let baseline_tetrahedron_node_ids = [[0, 3, 4, 5], [0, 4, 3, 6], [0, 5, 6, 3]];
    let baseline_tetrahedra = baseline_tetrahedron_node_ids
        .into_iter()
        .map(|node_ids| {
            raw_refill_tetrahedron_with_rejection_reason(
                node_ids,
                node_ids.map(|node_id| boundary_nodes[&node_id]),
                options,
            )
            .expect("baseline tetrahedron should pass fixture quality gates")
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for node_ids in baseline_tetrahedron_node_ids {
        for face in tetrahedron_faces(node_ids) {
            *face_counts.entry(sorted_face(face)).or_default() += 1;
        }
    }
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2, 3],
        boundary_faces: face_counts
            .into_iter()
            .filter_map(|(node_ids, count)| {
                (count == 1).then_some(ConstrainedCavityBoundaryFace {
                    node_ids,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: None,
                    source_edge_ids: [None, None, None],
                    region_ids: vec!["body".to_string()],
                })
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: baseline_tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.volume_m3)
            .sum(),
    };
    let baseline = refill_from_tetrahedra(
        &cavity,
        baseline_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("baseline should preserve the cavity boundary");

    let evaluation = evaluate_constrained_cavity_refill_candidates(&cavity, &nodes, &[], options)
        .expect("refill evaluation should complete");

    let refill = evaluation.refill.expect("boundary-node refill should pass");
    assert_eq!(refill.tetrahedra.len(), 2);
    assert!(refill_is_better(&refill, &baseline));
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 4, 5, 6]),
            sorted_tetrahedron_nodes([3, 4, 5, 6])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("flipped refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("flipped refill should preserve volume");
}

#[test]
fn boundary_node_exact_cover_supports_bounded_multi_ring_bipyramid() {
    let ring_count = 7_u32;
    let top_node_id = ring_count;
    let bottom_node_id = ring_count + 1;
    let mut nodes = (0..ring_count)
        .map(|node_id| {
            let angle = std::f64::consts::TAU * node_id as f64 / ring_count as f64;
            ConstrainedCavityNode {
                node_id,
                coordinates_m: [angle.cos(), angle.sin(), 0.0],
            }
        })
        .collect::<Vec<_>>();
    nodes.push(ConstrainedCavityNode {
        node_id: top_node_id,
        coordinates_m: [0.0, 0.0, 1.0],
    });
    nodes.push(ConstrainedCavityNode {
        node_id: bottom_node_id,
        coordinates_m: [0.0, 0.0, -1.0],
    });

    let options = refill_options();
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    let mut expected_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for node_id in 0..ring_count {
        let next_node_id = (node_id + 1) % ring_count;
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [top_node_id, node_id, next_node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [bottom_node_id, next_node_id, node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        let tetrahedron_node_ids = [top_node_id, bottom_node_id, node_id, next_node_id];
        expected_tetrahedra.push(
            raw_refill_tetrahedron_with_rejection_reason(
                tetrahedron_node_ids,
                tetrahedron_node_ids.map(|id| node_map[&id]),
                options,
            )
            .expect("ring bipyramid tetrahedron should pass quality gates"),
        );
    }
    let expected_volume_m3 = expected_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces,
        protected_node_ids: Vec::new(),
        target_volume_m3: expected_volume_m3,
    };
    validate_constrained_cavity(&cavity).expect("ring bipyramid cavity should validate");
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("bounded ring bipyramid should have an exact cover");

    assert_eq!(refill.tetrahedra.len(), ring_count as usize);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve the larger cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve the larger cavity volume");
}

#[test]
fn exact_cover_refill_selects_compatible_subset() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let candidate_nodes = [[0, 1, 2, 3], [0, 2, 1, 4], [1, 2, 3, 4]];
    let candidates = candidate_nodes
        .map(|node_ids| {
            let points = node_ids.map(|node_id| boundary_nodes[&node_id]);
            raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options)
                .expect("fixture tetrahedron should pass quality gates")
        })
        .to_vec();

    let refill = exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidates, options)
        .expect("exact cover refill should evaluate")
        .expect("exact cover should select the compatible subset");

    assert_eq!(refill.tetrahedra.len(), 2);
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            sorted_tetrahedron_nodes([0, 1, 2, 3]),
            sorted_tetrahedron_nodes([0, 2, 1, 4])
        ])
    );
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("selected subset should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("selected subset should preserve volume");
}

#[test]
fn exact_cover_on_demand_interior_mates_recovers_forced_mate() {
    let options = refill_options();
    let central = synthetic_refill_tetrahedron([0, 1, 2, 3], 1.0);
    let caps = [
        synthetic_refill_tetrahedron([0, 2, 1, 4], 1.0),
        synthetic_refill_tetrahedron([0, 1, 3, 5], 1.0),
        synthetic_refill_tetrahedron([0, 3, 2, 6], 1.0),
        synthetic_refill_tetrahedron([1, 2, 3, 7], 1.0),
    ];
    let shared_faces = BTreeSet::from([
        sorted_face([0, 1, 2]),
        sorted_face([0, 1, 3]),
        sorted_face([0, 2, 3]),
        sorted_face([1, 2, 3]),
    ]);
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: caps
            .iter()
            .flat_map(|tetrahedron| tetrahedron_faces(tetrahedron.node_ids))
            .map(sorted_face)
            .filter(|face| !shared_faces.contains(face))
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 5.0,
    };
    let refill = exact_cover_refill_from_on_demand_interior_mates(
        &cavity,
        caps.to_vec(),
        caps.into_iter().chain([central]).collect(),
        options,
    )
    .expect("on-demand exact cover should evaluate")
    .expect("on-demand mate injection should recover the cover");

    assert_eq!(refill.tetrahedra.len(), 5);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("on-demand exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("on-demand exact cover should preserve volume");
}

#[test]
fn exact_cover_refill_maximizes_worst_selected_quality() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: [
            [4, 0, 1],
            [4, 1, 2],
            [4, 2, 3],
            [4, 3, 0],
            [5, 1, 0],
            [5, 2, 1],
            [5, 3, 2],
            [5, 0, 3],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let low_worst_cover = [
        ([4, 5, 0, 1], 0.90),
        ([4, 5, 1, 2], 0.20),
        ([4, 5, 2, 3], 0.20),
        ([4, 5, 3, 0], 0.20),
    ];
    let better_worst_cover = [
        ([0, 2, 4, 1], 0.50),
        ([0, 2, 4, 3], 0.50),
        ([0, 2, 5, 1], 0.50),
        ([0, 2, 5, 3], 0.50),
    ];
    let candidates = low_worst_cover
        .into_iter()
        .chain(better_worst_cover)
        .map(
            |(node_ids, exact_scaled_jacobian)| ConstrainedCavityRefillTetrahedron {
                node_ids,
                volume_m3: 0.25,
                aspect_ratio: 1.0,
                exact_scaled_jacobian,
            },
        )
        .collect::<Vec<_>>();

    let refill = exact_cover_refill_from_candidate_tetrahedra(
        &cavity,
        &candidates,
        ConstrainedCavityRefillOptions {
            volume_relative_tolerance: 1.0e-9,
            ..refill_options()
        },
    )
    .expect("exact cover should evaluate")
    .expect("octahedron cavity should have a cover");

    assert_eq!(refill.tetrahedra.len(), 4);
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
            .collect::<BTreeSet<_>>(),
        better_worst_cover
            .into_iter()
            .map(|(node_ids, _)| sorted_tetrahedron_nodes(node_ids))
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(
        refill
            .tetrahedra
            .iter()
            .map(|tetrahedron| tetrahedron.exact_scaled_jacobian)
            .fold(f64::INFINITY, f64::min),
        0.50
    );
}

#[test]
fn exact_cover_root_availability_reports_boundary_face_candidates() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let candidates = [lower, upper];
    let search = BoundaryExactCoverSearch::new(
        &cavity,
        &candidates,
        refill_options().volume_relative_tolerance,
    );
    let availability = search.root_boundary_availability();

    assert_eq!(availability.zero_raw_candidate_face_count, 0);
    assert_eq!(availability.zero_addable_candidate_face_count, 0);
    assert!(availability.min_raw_candidate_count > 0);
    assert!(availability.min_addable_candidate_count > 0);
    assert!(availability.max_addable_candidate_count >= availability.min_addable_candidate_count);
}

#[test]
fn boundary_exact_cover_diagnostic_reports_relaxed_cover_feasibility() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_boundary_exact_cover(&cavity, &nodes, refill_options())
        .expect("diagnostic should evaluate");

    assert_eq!(diagnostic.boundary_node_count, 5);
    assert_eq!(diagnostic.boundary_face_count, 6);
    assert!(diagnostic.candidate_count > 0);
    assert_eq!(diagnostic.solid_candidate_count, diagnostic.candidate_count);
    assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
    assert_eq!(diagnostic.zero_solid_candidate_boundary_face_count, 0);
    assert!(diagnostic.min_boundary_face_candidate_count > 0);
    assert!(diagnostic.min_solid_boundary_face_candidate_count > 0);
    assert!(
        diagnostic.max_boundary_face_candidate_count
            >= diagnostic.min_boundary_face_candidate_count
    );
    assert!(
        diagnostic.max_solid_boundary_face_candidate_count
            >= diagnostic.min_solid_boundary_face_candidate_count
    );
    assert!(diagnostic.search_attempt_count > 0);
    assert!(diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "cover_found");
    assert_eq!(diagnostic.selected_tetrahedron_count, 2);
}

#[test]
fn exact_cover_search_targets_unpaired_interior_faces_after_boundary_faces() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 2],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
            ConstrainedCavityBoundaryFace {
                node_ids: [3, 4, 5],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
        ],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 6],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [3, 4, 5, 7],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 6, 8],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 8, 9],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [1, 6, 8, 10],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 6, 8, 11],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
    ];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for face in [[0, 1, 2], [3, 4, 5], sorted_face([0, 1, 6])] {
        face_counts.insert(face, 1);
    }

    let candidates = search
        .next_cover_candidates(&face_counts, &[0, 1])
        .expect("unpaired interior face should request connector candidates");

    assert_eq!(candidates, vec![2]);
}

#[test]
fn exact_cover_search_prunes_orphan_interior_faces() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![ConstrainedCavityBoundaryFace {
            node_ids: [0, 1, 2],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        }],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 4, 5],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [1, 2, 4, 6],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 4, 7],
            volume_m3: 0.2,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.3,
        },
    ];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let candidates = search
        .next_cover_candidates(&BTreeMap::new(), &[])
        .expect("boundary face should request cover candidates");

    assert_eq!(candidates, vec![1]);
}

#[test]
fn exact_cover_search_forces_single_interior_mate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 3, 4],
        volume_m3: 0.2,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let mut face_counts = BTreeMap::from([
        (sorted_face([0, 1, 3]), 1),
        (sorted_face([0, 1, 4]), 1),
        (sorted_face([0, 3, 4]), 1),
        (sorted_face([1, 3, 4]), 1),
    ]);
    let mut selected = Vec::<usize>::new();

    let propagated = search
        .propagate_forced_interior_mates(0.0, &mut face_counts, &mut selected)
        .expect("single interior mate should be forced");

    assert_eq!(propagated, (0.2, vec![0]));
    assert_eq!(selected, vec![0]);
    assert!(face_counts.values().all(|count| *count == 2));
}

#[test]
fn exact_cover_search_rolls_back_forced_mates_on_volume_failure() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 0.1,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 3, 4],
        volume_m3: 0.2,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([
        (sorted_face([0, 1, 3]), 1),
        (sorted_face([0, 1, 4]), 1),
        (sorted_face([0, 3, 4]), 1),
        (sorted_face([1, 3, 4]), 1),
    ]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();

    assert!(search
        .propagate_forced_interior_mates(0.0, &mut face_counts, &mut selected)
        .is_none());
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_without_addable_mate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [4, 5, 6, 7],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([(sorted_face([0, 1, 2]), 1)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::NoCandidateContainsFace
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_face_count_conflict() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts =
        BTreeMap::from([(sorted_face([0, 1, 2]), 1), (sorted_face([0, 1, 3]), 2)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::FaceCountConflict
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_future_mate_conflict() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([(sorted_face([0, 1, 2]), 1)]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::NoAddableMate {
            face: Some([0, 1, 2]),
            reason: ForcedInteriorMateNoAddableReason::FutureMateConflict
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_trace_reports_forced_mate_volume_overflow() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: Vec::new(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 0.1,
    };
    let candidates = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 3, 4],
        volume_m3: 0.2,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);
    let initial_face_counts = BTreeMap::from([
        (sorted_face([0, 1, 3]), 1),
        (sorted_face([0, 1, 4]), 1),
        (sorted_face([0, 3, 4]), 1),
        (sorted_face([1, 3, 4]), 1),
    ]);
    let mut face_counts = initial_face_counts.clone();
    let mut selected = Vec::<usize>::new();
    let mut selected_roles = Vec::<&'static str>::new();

    let result = search.propagate_forced_interior_mates_traced(
        0.0,
        &mut face_counts,
        &mut selected,
        &mut selected_roles,
    );

    assert_eq!(
        result,
        Err(ForcedInteriorMateFailure::VolumeOverflow {
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.2,
            target_volume_m3: 0.1,
        })
    );
    assert_eq!(face_counts, initial_face_counts);
    assert!(selected.is_empty());
    assert!(selected_roles.is_empty());
}

#[test]
fn exact_cover_face_candidate_source_diagnostic_reports_available_sources() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = diagnostic_boundary_exact_cover_face_candidate_sources(
        &cavity,
        &nodes,
        [0, 1, 3],
        refill_options(),
    )
    .expect("face candidate source diagnostic should evaluate");

    assert_eq!(diagnostic.target_face, [0, 1, 3]);
    assert_eq!(diagnostic.fourth_node_count, 2);
    assert_eq!(diagnostic.centroid_inside_count, 1);
    assert_eq!(diagnostic.solid_pass_count, 1);
    assert_eq!(diagnostic.relaxed_pass_count, 1);
    assert_eq!(diagnostic.outside_surface_count, 1);
    assert!(diagnostic.solid_rejected_by_reason.is_empty());
    assert!(diagnostic.relaxed_rejected_by_reason.is_empty());
    assert_eq!(diagnostic.relaxed_candidate_node_ids, vec![[0, 1, 2, 3]]);
}

#[test]
fn exact_cover_face_count_blockers_report_selected_blockers() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = selected_exact_cover_face_count_blockers(
        &cavity,
        &nodes,
        &[[0, 1, 2, 3], [0, 1, 2, 9]],
        [0, 1, 4],
        refill_options(),
    )
    .expect("face-count blocker diagnostic should evaluate");

    assert_eq!(diagnostic.target_face, [0, 1, 4]);
    assert_eq!(diagnostic.selected_tetrahedron_count, 2);
    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.blocker_count, 1);
    assert_eq!(diagnostic.blockers[0].node_ids, [0, 2, 1, 4]);
    assert!((diagnostic.blockers[0].exact_scaled_jacobian - 0.7071067811865475).abs() < 1.0e-15);
    assert_eq!(diagnostic.blockers[0].conflicting_faces, vec![[0, 1, 2]]);
    assert_eq!(
        diagnostic.blockers[0].blocking_selected_tetrahedra,
        vec![[0, 1, 2, 3], [0, 1, 2, 9]]
    );
}

#[test]
fn exact_cover_saturated_component_walks_selected_tetrahedron_component() {
    let cavity = two_tetrahedron_bipyramid_cavity();

    let diagnostic = selected_exact_cover_saturated_component(
        &cavity,
        &[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 9]],
        [0, 1, 2],
    );

    assert_eq!(diagnostic.seed_face, [0, 1, 2]);
    assert_eq!(diagnostic.saturated_face_count, 2);
    assert_eq!(diagnostic.component_face_count, 2);
    assert_eq!(diagnostic.component_tetrahedron_count, 3);
    assert_eq!(diagnostic.component_faces, vec![[0, 1, 2], [0, 1, 9]]);
    assert_eq!(
        diagnostic.component_tetrahedra,
        vec![[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 9]]
    );
}

#[test]
fn exact_cover_search_uses_configured_attempt_limit() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 1.0 / 6.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut low_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 1);

    assert!(low_limit_search.search().is_none());
    assert!(low_limit_search.attempts > 1);

    let mut sufficient_limit_search =
        BoundaryExactCoverSearch::with_attempt_limit(&cavity, &candidates, 1.0e-9, 2);

    assert_eq!(sufficient_limit_search.search(), Some(vec![0, 1]));
    assert_eq!(sufficient_limit_search.attempts, 2);
}

#[test]
fn exact_cover_trace_reports_volume_overflow_dead_end() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let candidates = [
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 3],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 2, 1, 4],
            volume_m3: 10.0,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.4,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-12);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "volume_overflow",
            face: None,
            depth: 1,
            selected_tetrahedra: vec![[0, 1, 2, 3]],
            selected_roles: vec!["branch"],
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0 / 3.0,
        })
    );
}

#[test]
fn boundary_steiner_exact_cover_diagnostic_reports_centroid_candidate_coverage() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic = diagnostic_boundary_steiner_exact_cover(&cavity, &nodes, refill_options())
        .expect("Steiner exact-cover diagnostic should evaluate");

    assert!(diagnostic.candidate_count > 0);
    assert_eq!(diagnostic.zero_candidate_boundary_face_count, 0);
    assert!(diagnostic.search_attempt_count > 0);
    assert_eq!(diagnostic.reason, "cover_found");
    assert!(diagnostic.selected_tetrahedron_count > 0);
}

#[test]
fn boundary_patch_steiner_exact_cover_diagnostic_reports_boundary_complete_fixture() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let diagnostic =
        diagnostic_boundary_patch_steiner_exact_cover(&cavity, &nodes, refill_options())
            .expect("patch Steiner exact-cover diagnostic should evaluate");

    assert_eq!(diagnostic.boundary_node_count, 8);
    assert_eq!(diagnostic.boundary_face_count, 12);
    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert_eq!(diagnostic.steiner_node_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert_eq!(diagnostic.search_attempt_count, 0);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn missing_face_local_cap_quality_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_quality(&cavity, &nodes, refill_options())
        .expect("local cap diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.pass_face_count, 0);
    assert_eq!(diagnostic.failed_face_count, 0);
    assert_eq!(diagnostic.candidate_count, 0);
    assert!(diagnostic.candidate_source_bins.is_empty());
    assert_eq!(diagnostic.max_scaled_jacobian, 0.0);
    assert_eq!(diagnostic.max_failed_face_scaled_jacobian, 0.0);
    assert!(diagnostic.failed_face_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.failed_face_source_bins.is_empty());
    assert!(diagnostic.rejected_by_reason.is_empty());
}

#[test]
fn local_cap_apex_candidates_include_optimized_normal_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.18, 0.72, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);

    let quality_for = |candidate: &LocalCapApexCandidate| {
        tetrahedron_scaled_jacobian([
            nodes[&face[0]],
            nodes[&face[1]],
            nodes[&face[2]],
            candidate.coordinates_m,
        ])
    };
    let best_discrete_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_discrete_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_positive = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_positive")
        .map(quality_for)
        .fold(0.0_f64, f64::max);
    let best_optimized_negative = candidates
        .iter()
        .filter(|candidate| candidate.source == "normal_optimized_negative")
        .map(quality_for)
        .fold(0.0_f64, f64::max);

    assert!(best_optimized_positive >= best_discrete_positive);
    assert!(best_optimized_negative >= best_discrete_negative);
}

#[test]
fn local_cap_apex_candidates_include_inplane_inward_offsets() {
    let face = [0, 1, 2];
    let nodes = BTreeMap::from([
        (0, [0.0, 0.0, 0.0]),
        (1, [1.0, 0.0, 0.0]),
        (2, [0.2, 0.8, 0.0]),
    ]);
    let surface_point = face_centroid(face, &nodes).expect("face should have a centroid");
    let candidates = local_cap_apex_candidates(face, surface_point, [0.3, 0.2, 0.8], &nodes);
    let inplane_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward")
        .collect::<Vec<_>>();
    let optimized_candidates = candidates
        .iter()
        .filter(|candidate| candidate.source == "inplane_inward_optimized")
        .collect::<Vec<_>>();

    assert!(!inplane_candidates.is_empty());
    assert!(!optimized_candidates.is_empty());
    assert!(inplane_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
    assert!(optimized_candidates.iter().any(|candidate| {
        candidate.coordinates_m[2] > surface_point[2]
            && (candidate.coordinates_m[0] - surface_point[0]).abs() > 1.0e-6
    }));
}

#[test]
fn missing_face_local_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic = diagnostic_missing_face_local_cap_stitch(&cavity, &nodes, refill_options())
        .expect("local cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.side_connector_candidate_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert_eq!(diagnostic.cap_side_face_count, 0);
    assert_eq!(diagnostic.zero_mate_cap_side_face_count, 0);
    assert_eq!(diagnostic.min_cap_side_face_mate_count, 0);
    assert_eq!(diagnostic.max_cap_side_face_mate_count, 0);
    assert_eq!(diagnostic.open_interior_face_count, 0);
    assert_eq!(diagnostic.open_interior_component_count, 0);
    assert!(diagnostic.open_interior_component_size_histogram.is_empty());
    assert_eq!(diagnostic.selected_tetrahedron_count, 0);
    assert_eq!(diagnostic.search_attempt_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
}

#[test]
fn missing_face_shared_patch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_shared_patch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("shared patch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.side_connector_candidate_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn missing_face_edge_subpatch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_edge_subpatch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("edge subpatch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn missing_face_hybrid_subpatch_cap_stitch_reports_boundary_complete_fixture() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let diagnostic =
        diagnostic_missing_face_hybrid_subpatch_cap_stitch(&cavity, &nodes, refill_options())
            .expect("hybrid subpatch cap stitch diagnostic should evaluate");

    assert_eq!(diagnostic.missing_face_count, 0);
    assert_eq!(diagnostic.patch_count, 0);
    assert!(diagnostic.patch_size_histogram.is_empty());
    assert!(diagnostic.patch_capped_face_count_histogram.is_empty());
    assert!(diagnostic.incomplete_patch_size_histogram.is_empty());
    assert_eq!(diagnostic.capped_face_count, 0);
    assert_eq!(diagnostic.inserted_node_count, 0);
    assert_eq!(diagnostic.candidate_tetrahedron_count, 0);
    assert!(!diagnostic.found_cover);
    assert_eq!(diagnostic.reason, "no_missing_faces");
}

#[test]
fn shared_patch_cap_finds_single_apex_for_simple_patch() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let node_coordinates = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let boundary_triangles = cavity_boundary_triangles(&cavity, &node_coordinates)
        .expect("unit tetrahedron boundary should be valid");
    let faces = [[0, 1, 2], [0, 1, 3]];

    let Some((coordinates_m, cap_tetrahedra)) = best_shared_patch_cap_for_faces(
        &faces,
        [0.25, 0.25, 0.25],
        4,
        &node_coordinates,
        &boundary_triangles,
        refill_options(),
    ) else {
        panic!("simple patch should have a shared cap apex");
    };

    assert_eq!(cap_tetrahedra.len(), faces.len());
    assert!(coordinates_m.iter().all(|value| value.is_finite()));
    assert!(cap_tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.node_ids.contains(&4)
            && tetrahedron.exact_scaled_jacobian.is_finite()));
}

#[test]
fn missing_face_components_separate_edge_and_node_connected_patches() {
    let faces = [[0, 1, 2], [2, 1, 3], [3, 4, 5], [3, 6, 7]];

    let edge_histogram =
        component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Edge));
    let node_histogram =
        component_size_histogram(missing_face_component_sizes(&faces, MissingFaceLink::Node));
    let node_components = missing_face_components(&faces, MissingFaceLink::Node);
    let common_node_ids =
        missing_face_component_common_node_ids(&faces, node_components.first().unwrap());

    assert_eq!(edge_histogram, BTreeMap::from([(1, 2), (2, 1)]));
    assert_eq!(node_histogram, BTreeMap::from([(4, 1)]));
    assert_eq!(common_node_ids, Vec::<u32>::new());

    let fan_faces = [[9, 1, 2], [9, 2, 3], [9, 3, 4]];
    let fan_components = missing_face_components(&fan_faces, MissingFaceLink::Node);
    assert_eq!(
        missing_face_component_common_node_ids(&fan_faces, fan_components.first().unwrap()),
        vec![9]
    );
}

#[test]
fn open_interior_refill_faces_reports_unpaired_non_boundary_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };

    assert_eq!(
        open_interior_refill_faces(&cavity, &[lower.clone()]),
        vec![[0, 1, 2]]
    );
    assert!(open_interior_refill_faces(&cavity, &[lower, upper]).is_empty());
}

#[test]
fn cap_side_face_mate_counts_report_connector_coverage() {
    let cap_tetrahedron = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 4],
        volume_m3: 1.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    };
    let mate_tetrahedron = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 4, 5],
        volume_m3: 1.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    };

    assert_eq!(
        cap_side_face_mate_counts(
            &[cap_tetrahedron.clone()],
            &[cap_tetrahedron, mate_tetrahedron],
            &BTreeSet::from([4])
        ),
        vec![1, 0, 0]
    );
}

#[test]
fn cap_side_connector_chain_adds_mates_for_open_inserted_faces() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = boundary_node_coordinates(&cavity, &two_tetrahedron_bipyramid_nodes())
        .expect("fixture nodes should cover cavity");
    nodes.insert(5, [0.25, 0.25, 0.0]);
    let boundary_triangles =
        cavity_boundary_triangles(&cavity, &nodes).expect("fixture boundary should evaluate");
    let mut candidate_tetrahedra = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 5],
        volume_m3: 0.1,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.5,
    }];
    let mut seen_tetrahedra = candidate_tetrahedra
        .iter()
        .map(|tetrahedron| sorted_tetrahedron_nodes(tetrahedron.node_ids))
        .collect::<BTreeSet<_>>();

    let inserted = append_cap_side_connector_chain_tetrahedra(
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &nodes,
        &BTreeSet::from([5]),
        &boundary_triangles,
        refill_options(),
    );

    assert!(inserted > 0);
    assert!(candidate_tetrahedra.len() > 1);
    assert!(candidate_tetrahedra
        .iter()
        .skip(1)
        .any(|tetrahedron| tetrahedron.node_ids.contains(&5)));
}

#[test]
fn cap_side_connector_chain_recovers_exact_cover_with_inserted_node_mates() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let mut nodes = boundary_node_coordinates(&cavity, &two_tetrahedron_bipyramid_nodes())
        .expect("fixture nodes should cover cavity");
    nodes.insert(5, [0.25, 0.25, 0.0]);
    let boundary_triangles =
        cavity_boundary_triangles(&cavity, &nodes).expect("fixture boundary should evaluate");
    let options = refill_options();
    let mut candidate_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for tetrahedron_node_ids in [[0, 1, 3, 5], [1, 2, 3, 5], [0, 2, 3, 5]] {
        let points = tetrahedron_node_ids.map(|node_id| nodes[&node_id]);
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(tetrahedron_node_ids, points, options)
        {
            seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids));
            candidate_tetrahedra.push(tetrahedron);
        }
    }
    assert!(
        exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidate_tetrahedra, options)
            .expect("initial exact cover should evaluate")
            .is_none()
    );
    let inserted = append_cap_side_connector_chain_tetrahedra(
        &mut candidate_tetrahedra,
        &mut seen_tetrahedra,
        &nodes,
        &BTreeSet::from([5]),
        &boundary_triangles,
        options,
    );
    assert_eq!(inserted, 3);
    let refill =
        exact_cover_refill_from_candidate_tetrahedra(&cavity, &candidate_tetrahedra, options)
            .expect("connector exact cover should evaluate")
            .expect("connector mates should close the inserted-node cover");
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("connector cover should preserve the cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("connector cover should preserve volume");
}

#[test]
fn candidate_orphan_interior_face_counts_report_global_orphans() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let lower = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 3],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };
    let upper = ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 2, 1, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 0.4,
    };

    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower.clone()]),
        (1, 0)
    );
    assert_eq!(
        candidate_orphan_interior_face_counts(&cavity, &[lower, upper]),
        (0, 2)
    );
}

#[test]
fn centroid_interior_refill_candidate_recovers_split_boundary_tetrahedron_cavity() {
    let mut cavity = unit_tetrahedron_cavity();
    let split_specs = [
        ([0, 2, 1], 4),
        ([0, 1, 3], 5),
        ([1, 2, 3], 6),
        ([2, 0, 3], 7),
    ];
    for (face, split_node_id) in split_specs {
        cavity.boundary_faces =
            split_constrained_cavity_boundary_faces(&cavity.boundary_faces, face, split_node_id)
                .expect("fixture face should split");
    }
    validate_constrained_cavity(&cavity).expect("split boundary fixture should be valid");
    let mut nodes = unit_tetrahedron_nodes();
    nodes.extend([
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [1.0 / 3.0, 0.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        },
        ConstrainedCavityNode {
            node_id: 7,
            coordinates_m: [0.0, 1.0 / 3.0, 1.0 / 3.0],
        },
    ]);

    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let refill = centroid_interior_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        refill_options(),
    )
    .expect("centroid interior refill should evaluate")
    .expect("centroid interior refill should recover the split boundary cavity");

    assert_eq!(refill.inserted_nodes.len(), 1);
    assert_eq!(refill.inserted_nodes[0].node_id, 8);
    assert_eq!(refill.tetrahedra.len(), cavity.boundary_faces.len());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("centroid interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        refill_options().volume_relative_tolerance,
    )
    .expect("centroid interior refill should preserve volume");
}

#[test]
fn interior_star_quality_diagnostic_bins_candidate_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let candidates = vec![
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [3.0, 3.0, 3.0],
        },
    ];

    let diagnostic = diagnostic_interior_star_quality(
        &cavity,
        &nodes,
        &candidates,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.01,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("interior star diagnostic should evaluate");

    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.pass_count, 1);
    assert!(diagnostic.max_min_scaled_jacobian >= 0.01);
    assert!(!diagnostic.min_scaled_jacobian_bins.is_empty());
    assert_eq!(
        diagnostic.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity", 1)])
    );
}

#[test]
fn two_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
    ];
    let options = refill_options();

    let refill = two_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("two-interior refill should evaluate")
    .expect("two-interior refill should recover the cavity");

    assert_eq!(refill.inserted_nodes, interior_candidates);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("two-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("two-interior refill should preserve volume");
}

#[test]
fn multi_interior_node_refill_preserves_bipyramid_cavity() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let interior_candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.25],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, -0.25],
        },
        ConstrainedCavityNode {
            node_id: 12,
            coordinates_m: [0.50, 0.25, 0.0],
        },
    ];
    let options = refill_options();

    let refill = multi_interior_node_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        &interior_candidates,
        options,
    )
    .expect("multi-interior refill should evaluate")
    .expect("multi-interior refill should recover the cavity");

    assert!(!refill.inserted_nodes.is_empty());
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("multi-interior refill should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("multi-interior refill should preserve volume");
}

#[test]
fn multi_interior_exact_cover_failure_reports_boundary_face_without_addable_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let options = refill_options();
    let lower_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let lower_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], lower_points, options)
            .expect("fixture tetrahedron should pass quality gates");

    assert_eq!(
        multi_interior_exact_cover_failure_reason(&cavity, &[lower_tetrahedron], options),
        "multi_interior_exact_cover_boundary_face_no_addable_candidate"
    );
}

#[test]
fn exact_cover_trace_reports_boundary_face_without_addable_candidate() {
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: vec![
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 2],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
            ConstrainedCavityBoundaryFace {
                node_ids: [0, 1, 3],
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            },
        ],
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    let candidates = vec![
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 2, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
        ConstrainedCavityRefillTetrahedron {
            node_ids: [0, 1, 3, 4],
            volume_m3: 0.1,
            aspect_ratio: 1.0,
            exact_scaled_jacobian: 0.5,
        },
    ];
    let mut search = BoundaryExactCoverSearch::new(&cavity, &candidates, 1.0e-9);

    let (selected, trace) = search.search_with_trace();

    assert!(selected.is_none());
    assert_eq!(
        trace.dead_end,
        Some(BoundaryExactCoverDeadEnd {
            reason: "boundary_face_no_addable_candidate",
            face: Some([0, 1, 2]),
            depth: 0,
            selected_tetrahedra: Vec::new(),
            selected_roles: Vec::new(),
            current_volume_m3: 0.0,
            candidate_volume_m3: 0.0,
            target_volume_m3: 1.0,
        })
    );
    assert_eq!(
        trace.dead_end_reason_counts,
        BTreeMap::from([("boundary_face_no_addable_candidate", 1)])
    );
    assert_eq!(
        trace.dead_end_faces_by_reason,
        BTreeMap::from([(
            "boundary_face_no_addable_candidate",
            BTreeSet::from([[0, 1, 2]])
        )])
    );
}

#[test]
fn boundary_face_completion_skips_duplicate_cap_tetrahedra() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_cap = raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], points, options)
        .expect("fixture cap should pass quality gates");

    let candidate = best_boundary_face_completion_tetrahedron(
        [0, 1, 2],
        &cavity,
        &boundary_nodes,
        &[duplicate_cap],
        &boundary_triangles,
        options,
    );

    assert!(candidate.is_none());
}

#[test]
fn boundary_face_completion_selector_reduces_boundary_delta() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();
    let duplicate_points = [0, 1, 2, 3].map(|node_id| boundary_nodes[&node_id]);
    let duplicate_tetrahedron =
        raw_refill_tetrahedron_with_rejection_reason([0, 1, 2, 3], duplicate_points, options)
            .expect("fixture duplicate should pass quality gates");
    let blocked_face = [0, 1, 2];
    let fillable_face = [0, 2, 4];

    let (selected_face, selected_tetrahedron) =
        best_boundary_face_completion_tetrahedron_for_faces(
            &[blocked_face, fillable_face],
            &cavity,
            &boundary_nodes,
            &[duplicate_tetrahedron.clone()],
            &boundary_triangles,
            options,
        )
        .expect("completion search should evaluate")
        .expect("completion search should find a delta-reducing face");

    let initial_delta = refill_boundary_face_delta(&cavity, &[duplicate_tetrahedron.clone()])
        .expect("initial delta should evaluate");
    let next_delta = refill_boundary_face_delta(
        &cavity,
        &[duplicate_tetrahedron, selected_tetrahedron.clone()],
    )
    .expect("next delta should evaluate");
    assert!(
        next_delta.missing.len() + next_delta.unexpected.len()
            < initial_delta.missing.len() + initial_delta.unexpected.len()
    );
    assert!(tetrahedron_faces(selected_tetrahedron.node_ids)
        .map(sorted_face)
        .contains(&sorted_face(selected_face)));
}

#[test]
fn refill_boundary_delta_reports_unexpected_faces() {
    let cavity = unit_tetrahedron_cavity();
    let refill_tetrahedra = vec![ConstrainedCavityRefillTetrahedron {
        node_ids: [0, 1, 2, 4],
        volume_m3: 1.0 / 6.0,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }];

    let delta = refill_boundary_face_delta(&cavity, &refill_tetrahedra)
        .expect("boundary delta should evaluate");

    assert!(delta.missing.contains(&[0, 1, 3]));
    assert!(delta.unexpected.contains(&[0, 1, 4]));
}

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
fn boundary_face_edge_split_completion_reports_inserted_node_and_refined_boundary() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");
    let options = refill_options();

    let (refined_cavity, inserted_node, split_tetrahedra) =
        best_boundary_face_edge_split_completion(
            [0, 1, 2],
            &cavity,
            &boundary_nodes,
            &boundary_triangles,
            &[],
            options,
        )
        .expect("edge-split completion should evaluate")
        .expect("edge-split completion should generate child cap tetrahedra");

    assert_eq!(inserted_node.node_id, 4);
    assert_eq!(inserted_node.coordinates_m[2], 0.0);
    assert!(
        (inserted_node.coordinates_m[0] == 0.0 && inserted_node.coordinates_m[1] > 0.0)
            || (inserted_node.coordinates_m[1] == 0.0 && inserted_node.coordinates_m[0] > 0.0)
            || (inserted_node.coordinates_m[0] + inserted_node.coordinates_m[1] - 1.0).abs()
                <= 1.0e-12
    );
    assert_eq!(split_tetrahedra.len(), 2);
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
        4
    );
    let refill = refill_from_tetrahedra(
        &refined_cavity,
        split_tetrahedra,
        options.volume_relative_tolerance,
    )
    .expect("edge-split child tetrahedra should preserve the refined boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("edge-split completion should preserve the original target volume");
}

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

#[test]
fn boundary_face_split_candidates_include_bounded_interior_lattice() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");

    let candidates = boundary_face_split_node_candidates([0, 1, 2], &boundary_nodes);

    assert!(candidates.len() >= 40);
    assert!(candidates.len() <= 64);
    assert!(candidates.iter().all(|node| node.node_id == 4));
    assert!(candidates.iter().all(|node| {
        node.coordinates_m[0] > 0.0
            && node.coordinates_m[1] > 0.0
            && node.coordinates_m[2] == 0.0
            && node.coordinates_m[0] + node.coordinates_m[1] < 1.0
    }));
    assert!(candidates.iter().any(|node| {
        (node.coordinates_m[0] - 0.1).abs() <= 1.0e-12
            && (node.coordinates_m[1] - 0.1).abs() <= 1.0e-12
    }));
}

#[test]
fn boundary_node_completion_diagnostic_classifies_no_cap_candidate() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();

    let diagnostic = diagnostic_boundary_node_completion(
        &cavity,
        &nodes,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.95,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("diagnostic should evaluate");

    assert_eq!(diagnostic.reason, "boundary_node_completion_no_candidate");
    assert!(diagnostic.missing_face_count > 0);
    assert_eq!(diagnostic.cap_candidate_count, 0);
    assert!(diagnostic.max_rejected_scaled_jacobian < 0.95);
    assert!(!diagnostic.rejected_scaled_jacobian_bins.is_empty());
    assert!(diagnostic.max_rejected_cap_height_ratio > 0.0);
    assert!(!diagnostic.rejected_cap_height_ratio_bins.is_empty());
    assert!(!diagnostic
        .rejected_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.rejected_cap_node_ids.is_empty());
    assert!(diagnostic.split_cap_candidate_count > 0);
    assert_eq!(diagnostic.split_cap_pass_count, 0);
    assert!(diagnostic.max_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic.edge_split_cap_scaled_jacobian_bins.is_empty());
    assert!(!diagnostic
        .edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic.edge_split_cap_apex_limited_node_ids.is_empty());
    assert!(diagnostic.three_edge_split_cap_candidate_count > 0);
    assert_eq!(diagnostic.three_edge_split_cap_pass_count, 0);
    assert!(diagnostic.max_three_edge_split_cap_scaled_jacobian < 0.95);
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_scaled_jacobian_worst_corner_bins
        .is_empty());
    assert!(!diagnostic
        .three_edge_split_cap_apex_limited_node_ids
        .is_empty());
    assert!(!diagnostic.rejected_by_reason.is_empty());
}

#[test]
fn refill_evaluation_skips_exterior_points_and_accepts_valid_candidate() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [2.0, 2.0, 2.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        refill_options(),
    )
    .expect("evaluation should complete");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity".to_string(), 1)])
    );
}

#[test]
fn refill_evaluation_skips_points_too_close_to_protected_boundary_nodes() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.01, 0.01, 0.01],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [0.25, 0.25, 0.25],
        },
    ];

    let evaluation = evaluate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect("evaluation should continue after protected-distance rejection");

    assert!(evaluation.refill.is_some());
    assert_eq!(
        evaluation.rejected_by_reason,
        BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
    );
}

#[test]
fn refill_generation_reports_protected_boundary_distance_rejections() {
    let mut cavity = unit_tetrahedron_cavity();
    cavity.protected_node_ids = vec![0];
    let nodes = unit_tetrahedron_nodes();
    let candidates = [ConstrainedCavityNode {
        node_id: 10,
        coordinates_m: [0.01, 0.01, 0.01],
    }];

    let err = generate_constrained_cavity_refill_candidates(
        &cavity,
        &nodes,
        &candidates,
        protected_refill_options(),
    )
    .expect_err("all candidates too close to protected nodes should fail");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::NoValidCandidate {
            rejected_by_reason: BTreeMap::from([("protected_boundary_distance".to_string(), 1)])
        }
    );
}

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

#[test]
fn star_refill_candidates_reject_boundary_node_reuse() {
    let cavity = unit_tetrahedron_cavity();
    let nodes = unit_tetrahedron_nodes();
    let reused = [ConstrainedCavityNode {
        node_id: 0,
        coordinates_m: [0.25, 0.25, 0.25],
    }];

    let err =
        generate_constrained_cavity_refill_candidates(&cavity, &nodes, &reused, refill_options())
            .expect_err("interior candidate cannot reuse a boundary node");

    assert_eq!(
        err,
        ConstrainedCavityRefillError::InteriorNodeReusesBoundaryNode { node_id: 0 }
    );
}

#[test]
fn validates_closed_tetrahedron_cavity_boundary() {
    let cavity = tetrahedron_cavity();

    let report = validate_constrained_cavity(&cavity).expect("closed cavity should validate");

    assert_eq!(report.boundary_face_count, 4);
    assert_eq!(report.boundary_edge_count, 6);
    assert_eq!(report.boundary_node_count, 4);
    assert_eq!(report.protected_node_count, 2);
    assert_eq!(report.target_volume_m3, 1.0);
}

#[test]
fn rejects_duplicate_boundary_faces() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[1].node_ids = cavity.boundary_faces[0].node_ids;

    let err =
        validate_constrained_cavity(&cavity).expect_err("duplicate boundary face should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::DuplicateBoundaryFace {
            node_ids: [0, 1, 2]
        }
    );
}

#[test]
fn rejects_open_boundary_edges() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces.pop();

    let err = validate_constrained_cavity(&cavity).expect_err("open boundary should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: 3
        }
    );
}

#[test]
fn rejects_protected_nodes_outside_boundary() {
    let mut cavity = tetrahedron_cavity();
    cavity.protected_node_ids.push(99);

    let err = validate_constrained_cavity(&cavity).expect_err("outside protected node should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary { node_id: 99 }
    );
}

#[test]
fn rejects_refill_volume_mismatch() {
    let err = validate_constrained_cavity_refill_volume(1.0, 1.2, 1.0e-9)
        .expect_err("volume mismatch should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3: 1.0,
            candidate_volume_m3: 1.2,
            tolerance_m3: 1.0e-9
        }
    );
}

#[test]
fn boundary_preservation_rejects_outside_neighbor_loss() {
    let mut cavity = tetrahedron_cavity();
    cavity.boundary_faces[0].outside_tetrahedron_ids = vec![99];
    let candidate_faces = cavity
        .boundary_faces
        .iter()
        .cloned()
        .map(|mut face| {
            if sorted_face(face.node_ids) == sorted_face(cavity.boundary_faces[0].node_ids) {
                face.outside_tetrahedron_ids.clear();
            }
            face
        })
        .collect::<Vec<_>>();

    let err = validate_constrained_cavity_boundary_preserved(&cavity, &candidate_faces)
        .expect_err("outside neighbor loss should fail");

    assert_eq!(
        err,
        ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch {
            node_ids: sorted_face(cavity.boundary_faces[0].node_ids),
            expected_outside_tetrahedron_ids: vec![99],
            candidate_outside_tetrahedron_ids: Vec::new(),
        }
    );
}

fn tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face([0, 1, 2]),
            face([0, 3, 1]),
            face([1, 3, 2]),
            face([2, 3, 0]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn face(node_ids: [u32; 3]) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: None,
        source_edge_ids: [None, None, None],
        region_ids: Vec::new(),
    }
}

fn provenance_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![7],
        boundary_faces: vec![
            face_with_provenance(
                [0, 1, 2],
                10,
                [Some(100), Some(101), Some(102)],
                &["loaded", "fixed"],
            ),
            face_with_provenance([0, 3, 1], 11, [Some(103), Some(104), Some(100)], &["fixed"]),
            face_with_provenance([1, 3, 2], 12, [Some(104), Some(105), Some(101)], &["solid"]),
            face_with_provenance([2, 3, 0], 13, [Some(105), Some(103), Some(102)], &["solid"]),
        ],
        protected_node_ids: vec![0, 1],
        target_volume_m3: 1.0,
    }
}

fn unit_tetrahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
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
        target_volume_m3: 1.0 / 6.0,
    }
}

fn unit_tetrahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
    ]
}

fn octahedron_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [1, 0, 5],
            [2, 1, 5],
            [3, 2, 5],
            [0, 3, 5],
        ]
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
        target_volume_m3: 4.0 / 3.0,
    }
}

fn octahedron_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [-1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, -1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn unit_cube_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1],
        boundary_faces: [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ]
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
        target_volume_m3: 1.0,
    }
}

fn unit_cube_nodes() -> Vec<ConstrainedCavityNode> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(node_id, coordinates_m)| ConstrainedCavityNode {
        node_id: node_id as u32,
        coordinates_m,
    })
    .collect()
}

fn two_tetrahedron_bipyramid_cavity() -> ConstrainedCavity {
    ConstrainedCavity {
        removed_tetrahedron_ids: vec![1, 2],
        boundary_faces: [
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            [0, 2, 4],
            [1, 2, 4],
            [0, 1, 4],
        ]
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
        target_volume_m3: 1.0 / 3.0,
    }
}

fn two_tetrahedron_bipyramid_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [0.0, 0.0, -1.0],
        },
    ]
}

fn two_tetrahedron_face_flip_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0 / 3.0, 1.0 / 3.0, -1.0],
        },
    ]
}

fn triangular_edge_ring_nodes() -> Vec<ConstrainedCavityNode> {
    vec![
        ConstrainedCavityNode {
            node_id: 0,
            coordinates_m: [0.0, 0.0, -1.0],
        },
        ConstrainedCavityNode {
            node_id: 3,
            coordinates_m: [0.0, 0.0, 1.0],
        },
        ConstrainedCavityNode {
            node_id: 4,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 5,
            coordinates_m: [-0.5, 0.8660254037844386, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 6,
            coordinates_m: [-0.5, -0.8660254037844386, 0.0],
        },
    ]
}

fn refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn protected_refill_options() -> ConstrainedCavityRefillOptions {
    ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        volume_relative_tolerance: 1.0e-12,
        min_protected_node_distance_m: 0.10,
        ..ConstrainedCavityRefillOptions::default()
    }
}

fn synthetic_refill_tetrahedron(
    node_ids: [u32; 4],
    volume_m3: f64,
) -> ConstrainedCavityRefillTetrahedron {
    ConstrainedCavityRefillTetrahedron {
        node_ids,
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}

fn face_with_provenance(
    node_ids: [u32; 3],
    source_face_id: u32,
    source_edge_ids: [Option<u32>; 3],
    region_ids: &[&str],
) -> ConstrainedCavityBoundaryFace {
    ConstrainedCavityBoundaryFace {
        node_ids,
        outside_tetrahedron_ids: Vec::new(),
        source_face_id: Some(source_face_id),
        source_edge_ids,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
    }
}

fn source_edge_for(face: &ConstrainedCavityBoundaryFace, edge: [u32; 2]) -> Option<u32> {
    face_edges(face.node_ids)
        .into_iter()
        .zip(face.source_edge_ids)
        .find_map(|(candidate_edge, source_edge_id)| {
            (sorted_edge(candidate_edge) == sorted_edge(edge)).then_some(source_edge_id)
        })
        .flatten()
}

fn candidate_tetrahedron(
    tetrahedron_id: u32,
    node_ids: [u32; 4],
    volume_m3: f64,
    region_ids: &[&str],
) -> CavityTetrahedron {
    CavityTetrahedron {
        tetrahedron_id,
        component_id: 0,
        node_ids,
        source_surface_element_id: 0,
        region_ids: region_ids.iter().map(|region| region.to_string()).collect(),
        volume_m3,
        aspect_ratio: 1.0,
        exact_scaled_jacobian: 1.0,
    }
}
