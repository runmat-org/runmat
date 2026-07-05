use super::*;
use fixtures::*;
use std::collections::BTreeMap;

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

#[test]
fn rejects_optimization_target_outcomes_that_exceed_target_seeds() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend.tetrahedron_optimization_target_seed_count = 2;
    mesh.backend
        .tetrahedron_optimization_skipped_target_seed_count = 1;
    mesh.backend.tetrahedron_optimization_rejected_edit_count = 2;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("optimization target outcomes cannot exceed target seeds");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_target_outcomes".to_string(),
            observed_count: 3,
            limit_count: 2,
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "inconsistent_tetrahedron_optimization_evidence"
    );
}

#[test]
fn rejects_optimization_edits_without_reported_pass() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_target_seed_count = 1;
    mesh.backend.tetrahedron_smoothed_point_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_accepted_count = 1;
    mesh.backend.tetrahedron_optimization_rejected_edit_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("optimization edits require an optimization, untangling, or repair pass");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_edits_without_pass".to_string(),
            observed_count: 2,
            limit_count: 0,
        }
    );
}

#[test]
fn rejects_interior_smoothing_outcomes_that_exceed_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_attempt_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_accepted_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_count = 1;
    mesh.backend.tetrahedron_smoothed_point_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 1)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("interior smoothing outcomes cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_interior_smoothing_outcomes".to_string(),
            observed_count: 2,
            limit_count: 1,
        }
    );
}

#[test]
fn rejects_interior_smoothing_accepted_count_mismatch() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_attempt_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_accepted_count = 1;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("smoothed point count must match accepted interior smoothing edits");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_interior_smoothing_accepted_points".to_string(),
            observed_count: 0,
            limit_count: 1,
        }
    );
}

#[test]
fn rejects_interior_smoothing_rejection_reason_mismatch() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_attempt_count = 2;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_count = 2;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 1)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("interior smoothing rejection reasons must reconcile");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_interior_smoothing_rejection_reasons".to_string(),
            observed_count: 1,
            limit_count: 2,
        }
    );
}

#[test]
fn rejects_interior_smoothing_attempts_without_reported_pass() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_attempt_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_count = 1;
    mesh.backend
        .tetrahedron_optimization_interior_smoothing_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 1)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("interior smoothing attempts require a reported repair pass");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_interior_smoothing_without_pass".to_string(),
            observed_count: 1,
            limit_count: 0,
        }
    );
}

#[test]
fn allows_skipped_optimization_targets_without_reported_pass() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_target_seed_count = 2;
    mesh.backend
        .tetrahedron_optimization_skipped_target_seed_count = 2;

    validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect("skipped targets are bookkeeping when no optimization pass ran");
}

#[test]
fn rejects_local_reconnection_rejections_that_exceed_attempts() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_attempt_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_accepted_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_count = 2;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 2)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("local reconnection outcomes cannot exceed attempts");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_local_reconnection_outcomes".to_string(),
            observed_count: 3,
            limit_count: 1,
        }
    );
}

#[test]
fn rejects_local_reconnection_rejection_reason_mismatch() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend.tetrahedron_optimization_pass_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_attempt_count = 3;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_count = 2;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 1)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("local reconnection rejection reasons must reconcile");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_local_reconnection_rejection_reasons".to_string(),
            observed_count: 1,
            limit_count: 2,
        }
    );
}

#[test]
fn rejects_local_reconnection_attempts_without_reported_pass() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_optimization_local_reconnection_attempt_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_count = 1;
    mesh.backend
        .tetrahedron_optimization_local_reconnection_rejected_by_reason =
        BTreeMap::from([("quality_does_not_improve".to_string(), 1)]);

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("local reconnection attempts require a reported repair pass");

    assert_eq!(
        err,
        AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
            family: "optimization_local_reconnections_without_pass".to_string(),
            observed_count: 1,
            limit_count: 0,
        }
    );
}

#[test]
fn rejects_reported_optimization_min_exact_quality_regression() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_optimization_initial_min_exact_scaled_jacobian = 0.40;
    mesh.backend
        .tetrahedron_optimization_final_min_exact_scaled_jacobian = 0.30;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("reported optimization quality cannot regress");

    assert_eq!(
        err,
        AnalysisMeshValidationError::TetrahedronOptimizationQualityRegression {
            metric: "min_exact_scaled_jacobian".to_string(),
            initial_value: "4.000000000000e-1".to_string(),
            final_value: "3.000000000000e-1".to_string(),
        }
    );
    assert_eq!(
        analysis_mesh_validation_error_code(&err),
        "tetrahedron_optimization_quality_regression"
    );
}

#[test]
fn rejects_reported_optimization_max_aspect_ratio_regression() {
    let mut mesh = valid_tetrahedron_mesh();
    mesh.backend
        .tetrahedron_optimization_initial_max_aspect_ratio = 4.0;
    mesh.backend.tetrahedron_optimization_final_max_aspect_ratio = 6.0;

    let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
        .expect_err("reported optimization aspect ratio cannot regress");

    assert_eq!(
        err,
        AnalysisMeshValidationError::TetrahedronOptimizationQualityRegression {
            metric: "max_aspect_ratio".to_string(),
            initial_value: "4.000000000000e0".to_string(),
            final_value: "6.000000000000e0".to_string(),
        }
    );
}
