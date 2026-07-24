use super::super::*;

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
