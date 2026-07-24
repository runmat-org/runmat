use super::*;

#[test]
fn rejects_boundary_leak_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_attempted_boundary_leak_recovery_item_count = 2;
    mesh.backend.tetrahedron_exposed_interior_source_face_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("boundary-leak outcomes must account for every attempted source face");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "boundary_leak_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_boundary_leak_rejection_reason_count_that_does_not_match_rejections() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_attempted_boundary_leak_recovery_item_count = 2;
    mesh.backend.tetrahedron_exposed_interior_source_face_count = 1;
    mesh.backend
        .tetrahedron_rejected_boundary_leak_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("boundary-leak rejection reasons must match rejected source faces");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "boundary_leak_rejection_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}
