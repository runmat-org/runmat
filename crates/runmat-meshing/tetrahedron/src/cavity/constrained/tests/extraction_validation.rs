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
