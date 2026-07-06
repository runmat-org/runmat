use super::*;

#[test]
fn rejects_attempted_volume_face_source_face_restoration_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted source-face boundary restoration cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_volume_face_source_face_boundary_restoration".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_volume_face_source_face_boundary_restoration_status_count_that_does_not_match_attempts()
{
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_volume_face_source_face_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "volume-face source-face boundary restoration outcomes must account for every attempt",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "volume_face_source_face_boundary_restoration_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_volume_face_source_face_boundary_restoration_rejection_reason_mismatch() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_volume_face_source_face_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_volume_face_source_face_boundary_restoration_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_volume_face_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_volume_face_source_face_boundary_restoration_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("source-face boundary restoration rejection reasons must match rejections");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "volume_face_source_face_boundary_restoration_rejection_reason_items"
                .to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}

#[test]
fn rejects_source_face_diagonal_pair_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_absent_face_source_face_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_source_face_diagonal_recovery_pair_count = 2;
    mesh.backend
        .tetrahedron_recovered_source_face_diagonal_pair_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("source-face diagonal pair outcomes must account for every attempted pair");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "source_face_diagonal_pair_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_attempted_material_interface_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_material_interface_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted material-interface evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_material_interface".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_material_interface_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_material_interface_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_material_interface_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("material-interface outcomes must account for every attempt");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "material_interface_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_material_interface_rejection_reason_count_that_does_not_match_rejections() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_material_interface_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_material_interface_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("material-interface rejection reasons must match rejected items");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "material_interface_rejection_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 0,
        }
    );
}

#[test]
fn rejects_attempted_absent_material_partition_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_attempted_absent_material_partition_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("attempted absent material partition evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "attempted_absent_material_partition".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_absent_material_partition_status_count_that_does_not_match_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_absent_material_partition_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_inserted_absent_material_partition_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_rejected_absent_material_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_absent_material_partition_quality_gate_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("absent partition inserted plus rejected outcomes must match attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "absent_material_partition_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 3,
        }
    );
}

#[test]
fn rejects_absent_material_partition_rejection_reason_count_that_does_not_match_rejections() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_missing_material_interface_absent_partition_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_attempted_absent_material_partition_recovery_item_count = 2;
    mesh.backend
        .tetrahedron_inserted_absent_material_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_absent_material_partition_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_rejected_absent_material_partition_facet_topology_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("absent partition rejection reasons must match rejected outcomes");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "absent_material_partition_rejection_reason_items".to_string(),
            aggregate_count: 1,
            typed_count: 2,
        }
    );
}
