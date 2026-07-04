use super::*;
use fixtures::*;

#[test]
fn rejects_solid_mesh_without_plc_input_evidence() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.backend = "solid".to_string();
    mesh.backend.algorithm = "plc_tetrahedron/v1".to_string();

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("solid Tetrahedron artifacts must prove PLC input evidence");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "missing_plc_nodes".to_string(),
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "missing_plc_input_evidence"
    );
}

#[test]
fn accepts_solid_mesh_with_classified_plc_input_evidence() {
    let mesh = solid_tetrahedron_mesh_with_plc_input_evidence();

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("classified PLC input evidence should satisfy solid validation");
}

#[test]
fn rejects_solid_mesh_with_unclassified_plc_shell_evidence() {
    let mut mesh = solid_tetrahedron_mesh_with_plc_input_evidence();
    mesh.backend.plc_input_shell_nesting_classified = false;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("unclassified PLC shell evidence should fail");

    assert_eq!(
        err,
        AnalysisMeshValidationError::MissingPlcInputEvidence {
            reason: "unclassified_plc_shell_nesting".to_string(),
        }
    );
}
