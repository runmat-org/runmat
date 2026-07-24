use super::*;

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
