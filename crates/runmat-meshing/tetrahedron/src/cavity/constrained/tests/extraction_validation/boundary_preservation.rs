use super::super::*;

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
