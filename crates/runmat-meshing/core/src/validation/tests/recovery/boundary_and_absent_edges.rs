use super::*;

#[test]
fn rejects_cad_curve_source_edge_status_count_that_does_not_match_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_cad_curve_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve source-edge status counts must reconcile with input count");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_source_edge_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_source_edge_missing_reason_count_that_does_not_match_missing_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_cad_curve_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve source-edge missing reasons must reconcile with missing count");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_source_edge_missing_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}

#[test]
fn rejects_recovered_source_face_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_volume_face_source_face_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered source-face evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "volume_face_source_face".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_recovered_absent_material_partition_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_absent_partition_material_interface_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered absent material partition evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "absent_partition_material_interface".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_absent_source_edge_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted absent source-edge evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_item_evidence"
    );
}

#[test]
fn rejects_cad_curve_absent_source_edge_reconnection_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "reconnected CAD curve absent source-edge evidence cannot exceed CAD-backed attempts",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "reconnected_cad_curve_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn allows_two_protected_edge_boundary_face_restoration_attempts_per_volume_edge_input() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_protected_edge_boundary_face_count = 2;

    validate_analysis_mesh(&mesh, Default::default())
        .expect("one manifold protected-edge input can restore two adjacent boundary faces");
}

#[test]
fn rejects_protected_edge_boundary_face_restoration_attempts_beyond_manifold_edge_bound() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 3;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("protected-edge boundary restoration attempts are bounded by adjacent facets");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_protected_edge_boundary_face_restoration".to_string(),
            item_count: 3,
            input_count: 2,
        }
    );
}

#[test]
fn rejects_rejected_protected_edge_boundary_face_restoration_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count = 3;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("rejected protected-edge restoration count cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "rejected_protected_edge_boundary_face_restoration".to_string(),
            item_count: 3,
            input_count: 2,
        }
    );
}

#[test]
fn rejects_recovered_protected_edge_boundary_face_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_protected_edge_boundary_face_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered protected-edge boundary-face evidence cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "recovered_protected_edge_boundary_face".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_protected_edge_boundary_face_restoration_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "recovered CAD curve protected-edge boundary-face evidence cannot exceed CAD attempts",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "recovered_cad_curve_protected_edge_boundary_face".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_protected_edge_boundary_face_restoration_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_protected_edge_boundary_face_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "protected-edge boundary-face restoration outcomes must account for every attempt",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "protected_edge_boundary_face_restoration_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_protected_edge_boundary_face_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_protected_edge_boundary_face_count = 2;
    mesh.backend
        .tetrahedron_attempted_cad_curve_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_cad_curve_protected_edge_boundary_face_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "CAD curve protected-edge boundary-face outcomes must account for every CAD attempt",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_protected_edge_boundary_face_restoration_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_protected_edge_boundary_face_rejection_reason_count_that_does_not_match_rejections() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_protected_edge_boundary_face_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_protected_edge_boundary_face_count = 1;
    mesh.backend
        .tetrahedron_rejected_protected_edge_boundary_face_restoration_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("protected-edge boundary-face rejection reasons must match rejections");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "protected_edge_boundary_face_restoration_rejection_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}

#[test]
fn rejects_rejected_absent_source_edge_count_that_exceeds_attempted_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_absent_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("rejected absent source-edge evidence cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "rejected_absent_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_absent_source_edge_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_reconnected_absent_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("absent source-edge outcomes must account for every attempt");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "absent_source_edge_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_absent_source_edge_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_edge_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_absent_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_cad_curve_absent_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_reconnected_absent_source_edge_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_reconnected_cad_curve_absent_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve absent source-edge outcomes must account for every CAD attempt");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "cad_curve_absent_source_edge_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}
