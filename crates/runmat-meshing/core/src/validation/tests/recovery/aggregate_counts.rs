use super::*;

#[test]
fn rejects_incomplete_tetrahedron_recovery() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 3;
    mesh.backend.tetrahedron_missing_recovery_item_count = 3;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_material_interface_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("remaining Tetrahedron recovery items should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::IncompleteTetrahedronRecoveryPresent {
            missing_item_count: 3,
            missing_source_face_item_count: 1,
            missing_source_edge_item_count: 1,
            missing_material_interface_item_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "incomplete_tetrahedron_recovery_present"
    );
}

#[test]
fn rejects_incomplete_tetrahedron_recovery_from_aggregate_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_missing_recovery_item_count = 4;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;

    let err = validate_analysis_mesh_with_options(&mesh, AnalysisMeshValidationOptions::default())
        .expect_err("inconsistent aggregate missing recovery evidence should fail readiness");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "missing_items".to_string(),
            aggregate_count: 4,
            typed_count: 1,
        }
    );
}

#[test]
fn rejects_missing_source_face_reason_count_that_exceeds_missing_source_faces() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 1;
    mesh.backend.tetrahedron_missing_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_topology_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("missing source-face reason counts must be bounded by missing source faces");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "missing_source_face_topology".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_missing_source_edge_reason_count_that_exceeds_missing_source_edges() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_edge_recovery_item_count = 1;
    mesh.backend.tetrahedron_missing_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_edge_absent_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("missing source-edge reason counts must be bounded by missing source edges");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "missing_source_edge_absent_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_missing_material_interface_reason_count_that_exceeds_missing_interfaces() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_material_interface_recovery_item_count = 1;
    mesh.backend.tetrahedron_missing_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_material_interface_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_material_interface_interior_face_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default()).expect_err(
        "missing material-interface reason counts must be bounded by missing interfaces",
    );

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "missing_material_interface_interior_face".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_recovery_item_aggregate_that_does_not_match_typed_inputs() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 1;
    mesh.backend.tetrahedron_source_edge_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("aggregate recovery items must match typed recovery inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "recovery_items".to_string(),
            aggregate_count: 1,
            typed_count: 2,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_aggregate_evidence"
    );
}

#[test]
fn rejects_recovery_status_aggregate_that_does_not_match_total_items() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_recovery_item_count = 2;
    mesh.backend.tetrahedron_source_face_recovery_item_count = 2;
    mesh.backend.tetrahedron_recovered_item_count = 2;
    mesh.backend
        .tetrahedron_recovered_source_face_recovery_item_count = 2;
    mesh.backend.tetrahedron_missing_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_missing_source_face_recovery_item_count = 1;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered plus missing recovery statuses must match total items");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryAggregateEvidence {
            family: "recovery_status_items".to_string(),
            aggregate_count: 2,
            typed_count: 3,
        }
    );
}
