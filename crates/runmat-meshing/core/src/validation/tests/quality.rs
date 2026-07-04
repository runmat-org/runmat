use super::*;
use fixtures::*;

#[test]
fn rejects_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.min_scaled_jacobian = 0.01;
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low jacobian should fail");
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "quality_threshold_failed"
    );
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_exact_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.min_exact_scaled_jacobian = 0.01;
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low exact jacobian should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_exact_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_element_exact_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.elements.push(ElementQuality {
        element_id: "e1".to_string(),
        scaled_jacobian: 0.8,
        exact_scaled_jacobian: 0.01,
        aspect_ratio: 1.0,
        volume_m3: 1.0 / 6.0,
    });
    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("low element exact jacobian should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "element_exact_scaled_jacobian".to_string()
        }
    );
}

#[test]
fn rejects_boundary_projection_quality_threshold_failures() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.quality.max_boundary_projection_error_m = 2.0e-6;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("boundary projection error should fail");

    assert_eq!(
        err,
        AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_boundary_projection_error_m".to_string()
        }
    );
}
