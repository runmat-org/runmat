use super::*;
use fixtures::*;

#[test]
fn rejects_mesh_that_underfills_expected_bounds() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_bounds_m: Some([[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]]),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail bounds coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundsCoverageFailed {
            axis: 0,
            coverage_ratio: "0.250000".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}

#[test]
fn rejects_mesh_that_underfills_expected_volume() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_volume_m3: Some(1.0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail volume coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::VolumeCoverageFailed {
            coverage_ratio: "0.166667".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}

#[test]
fn rejects_uncovered_interior_coverage_samples() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("uncovered interior coverage sample should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: "0.500000".to_string(),
            required_ratio: "1.000000".to_string(),
        }
    );
}

#[test]
fn accepts_covered_interior_coverage_samples() {
    let mesh = valid_tetrahedron_mesh();
    validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect("covered interior coverage sample should pass");
}

#[test]
fn rejects_nearby_uncovered_samples_for_small_tetrahedra() {
    let mut mesh = valid_tetrahedron_mesh();
    for node in &mut mesh.nodes {
        for coordinate in &mut node.coordinates_m {
            *coordinate *= 1.0e-3;
        }
    }
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            coverage_sample_points_m: vec![[1.01e-3, 1.0e-6, 1.0e-6]],
            min_coverage_sample_ratio: 1.0,
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("sample outside a small tetrahedron should fail");
    assert_eq!(
        err,
        AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: "0.000000".to_string(),
            required_ratio: "1.000000".to_string(),
        }
    );
}

#[test]
fn rejects_mesh_that_underfills_expected_boundary_area() {
    let mesh = valid_tetrahedron_mesh();
    let err = validate_analysis_mesh_with_options(
        &mesh,
        AnalysisMeshValidationOptions {
            expected_boundary_area_m2: Some(2.0),
            ..AnalysisMeshValidationOptions::default()
        },
    )
    .expect_err("mesh should fail boundary area coverage");
    assert_eq!(
        err,
        AnalysisMeshValidationError::BoundaryAreaCoverageFailed {
            area_ratio: "0.250000".to_string(),
            required_ratio: "0.900000".to_string(),
        }
    );
}
