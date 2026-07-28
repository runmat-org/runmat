use crate::contracts::AnalysisMeshArtifact;

use super::{
    geometry::{mesh_boundary_area_m2, mesh_bounds_m, mesh_contains_point, mesh_volume_m3},
    AnalysisMeshValidationError,
};

pub(super) fn validate_coverage_samples(
    mesh: &AnalysisMeshArtifact,
    coverage_sample_points_m: &[[f64; 3]],
    min_coverage_sample_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    if coverage_sample_points_m.is_empty()
        || !min_coverage_sample_ratio.is_finite()
        || min_coverage_sample_ratio <= 0.0
    {
        return Ok(());
    }
    let finite_samples = coverage_sample_points_m
        .iter()
        .copied()
        .filter(|point| point.iter().all(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if finite_samples.is_empty() {
        return Ok(());
    }
    let covered_count = finite_samples
        .iter()
        .filter(|point| mesh_contains_point(mesh, **point))
        .count();
    let coverage_ratio = covered_count as f64 / finite_samples.len() as f64;
    if coverage_ratio + 1.0e-9 < min_coverage_sample_ratio {
        return Err(AnalysisMeshValidationError::CoverageSampleFailed {
            coverage_ratio: format!("{coverage_ratio:.6}"),
            required_ratio: format!("{min_coverage_sample_ratio:.6}"),
        });
    }
    Ok(())
}

pub(super) fn validate_bounds_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_bounds_m: Option<[[f64; 3]; 2]>,
    min_bounds_coverage_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected) = expected_bounds_m else {
        return Ok(());
    };
    if !min_bounds_coverage_ratio.is_finite() || min_bounds_coverage_ratio <= 0.0 {
        return Ok(());
    }
    let Some(actual) = mesh_bounds_m(mesh) else {
        return Ok(());
    };
    for axis in 0..3 {
        let expected_min = expected[0][axis].min(expected[1][axis]);
        let expected_max = expected[0][axis].max(expected[1][axis]);
        if !expected_min.is_finite() || !expected_max.is_finite() {
            continue;
        }
        let expected_span = expected_max - expected_min;
        if expected_span <= f64::EPSILON {
            continue;
        }
        let actual_min = actual[0][axis].min(actual[1][axis]);
        let actual_max = actual[0][axis].max(actual[1][axis]);
        let overlap = (actual_max.min(expected_max) - actual_min.max(expected_min)).max(0.0);
        let coverage = overlap / expected_span;
        if coverage + 1.0e-9 < min_bounds_coverage_ratio {
            return Err(AnalysisMeshValidationError::BoundsCoverageFailed {
                axis,
                coverage_ratio: format!("{coverage:.6}"),
                required_ratio: format!("{min_bounds_coverage_ratio:.6}"),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_volume_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_volume_m3: Option<f64>,
    min_volume_coverage_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected_volume_m3) = expected_volume_m3 else {
        return Ok(());
    };
    if !expected_volume_m3.is_finite()
        || expected_volume_m3 <= f64::EPSILON
        || !min_volume_coverage_ratio.is_finite()
        || min_volume_coverage_ratio <= 0.0
    {
        return Ok(());
    }
    let actual_volume_m3 = mesh_volume_m3(mesh);
    let coverage_ratio = actual_volume_m3 / expected_volume_m3;
    if coverage_ratio + 1.0e-9 < min_volume_coverage_ratio
        || coverage_ratio - 1.0e-9 > 1.0 / min_volume_coverage_ratio
    {
        return Err(AnalysisMeshValidationError::VolumeCoverageFailed {
            coverage_ratio: format!("{coverage_ratio:.6}"),
            required_ratio: format!("{min_volume_coverage_ratio:.6}"),
        });
    }
    Ok(())
}

pub(super) fn validate_boundary_area_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_boundary_area_m2: Option<f64>,
    min_boundary_area_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected_boundary_area_m2) = expected_boundary_area_m2 else {
        return Ok(());
    };
    if !expected_boundary_area_m2.is_finite()
        || expected_boundary_area_m2 <= f64::EPSILON
        || !min_boundary_area_ratio.is_finite()
        || min_boundary_area_ratio <= 0.0
    {
        return Ok(());
    }
    let actual_boundary_area_m2 = mesh_boundary_area_m2(mesh);
    let area_ratio = actual_boundary_area_m2 / expected_boundary_area_m2;
    if area_ratio + 1.0e-9 < min_boundary_area_ratio
        || area_ratio - 1.0e-9 > 1.0 / min_boundary_area_ratio
    {
        return Err(AnalysisMeshValidationError::BoundaryAreaCoverageFailed {
            area_ratio: format!("{area_ratio:.6}"),
            required_ratio: format!("{min_boundary_area_ratio:.6}"),
        });
    }
    Ok(())
}
