use super::*;

#[test]
fn rejects_recovered_source_edge_count_that_exceeds_typed_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_boundary_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_boundary_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered source-edge evidence cannot exceed typed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "boundary_edge_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_recovery_evidence"
    );
}

#[test]
fn rejects_recovered_cad_curve_source_edge_count_that_exceeds_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_cad_curve_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered CAD curve source-edge evidence cannot exceed CAD-backed inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "cad_curve_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_cad_curve_interior_edge_source_edge_count_that_exceeds_interior_edge_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("CAD curve interior-edge evidence cannot exceed interior-edge inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryItemEvidence {
            family: "cad_curve_interior_edge_source_edge".to_string(),
            item_count: 2,
            input_count: 1,
        }
    );
}

#[test]
fn rejects_recovered_cad_curve_interior_edge_source_edge_count_that_exceeds_cad_input_count() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_cad_curve_interior_edge_source_edge_recovery_item_count = 1;
    mesh.backend
        .tetrahedron_recovered_cad_curve_interior_edge_source_edge_recovery_item_count = 2;

    let err = validate_analysis_mesh(&mesh, Default::default())
        .expect_err("recovered CAD curve interior-edge evidence cannot exceed CAD inputs");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronRecoveryEvidence {
            family: "cad_curve_interior_edge_source_edge".to_string(),
            recovered_count: 2,
            input_count: 1,
        }
    );
}
