use crate::{contracts::AnalysisMeshArtifact, quality::QualityThresholds};

use super::AnalysisMeshValidationError;

pub(super) fn validate_quality(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    if !mesh.quality.min_scaled_jacobian.is_finite()
        || mesh.quality.min_scaled_jacobian < thresholds.min_scaled_jacobian
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_scaled_jacobian".to_string(),
        });
    }
    if !mesh.quality.min_exact_scaled_jacobian.is_finite()
        || mesh.quality.min_exact_scaled_jacobian < thresholds.min_scaled_jacobian
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_exact_scaled_jacobian".to_string(),
        });
    }
    if mesh.quality.elements.iter().any(|element| {
        !element.exact_scaled_jacobian.is_finite()
            || element.exact_scaled_jacobian < thresholds.min_scaled_jacobian
    }) {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "element_exact_scaled_jacobian".to_string(),
        });
    }
    if !mesh.quality.max_aspect_ratio.is_finite()
        || mesh.quality.max_aspect_ratio > thresholds.max_aspect_ratio
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_aspect_ratio".to_string(),
        });
    }
    if !mesh.quality.max_boundary_projection_error_m.is_finite()
        || mesh.quality.max_boundary_projection_error_m > thresholds.max_boundary_projection_error_m
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_boundary_projection_error_m".to_string(),
        });
    }
    if !thresholds.allow_inverted_elements && mesh.quality.inverted_element_count > 0 {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "inverted_element_count".to_string(),
        });
    }
    validate_optimization_evidence(mesh)?;
    Ok(())
}

fn validate_optimization_evidence(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), AnalysisMeshValidationError> {
    let backend = &mesh.backend;
    let completed_target_count = backend.tetrahedron_optimization_skipped_target_seed_count
        + backend.tetrahedron_optimization_rejected_edit_count;
    if completed_target_count > backend.tetrahedron_optimization_target_seed_count {
        return Err(
            AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
                family: "optimization_target_outcomes".to_string(),
                observed_count: completed_target_count,
                limit_count: backend.tetrahedron_optimization_target_seed_count,
            },
        );
    }

    let reported_edit_count = backend.tetrahedron_smoothed_point_count
        + backend.tetrahedron_sliver_removed_count
        + backend.tetrahedron_optimization_rejected_edit_count
        + backend.tetrahedron_untangling_relocated_seed_count
        + backend.tetrahedron_untangling_reconnected_edge_star_count
        + backend.tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count
        + backend.tetrahedron_untangling_reconnected_node_adjacent_cavity_count
        + backend.tetrahedron_exact_quality_reconnected_cavity_count
        + backend.tetrahedron_exact_quality_reconnection_quality_gain_count;
    let repair_pass_count = backend.tetrahedron_optimization_pass_count
        + backend.tetrahedron_untangling_pass_count
        + backend.tetrahedron_exact_quality_repair_pass_count;
    if repair_pass_count == 0 && reported_edit_count > 0 {
        return Err(
            AnalysisMeshValidationError::InconsistentTetrahedronOptimizationEvidence {
                family: "optimization_edits_without_pass".to_string(),
                observed_count: reported_edit_count,
                limit_count: repair_pass_count,
            },
        );
    }

    let initial_min = backend.tetrahedron_optimization_initial_min_exact_scaled_jacobian;
    let final_min = backend.tetrahedron_optimization_final_min_exact_scaled_jacobian;
    if initial_min.is_finite()
        && final_min.is_finite()
        && initial_min > 0.0
        && final_min + 1.0e-12 < initial_min
    {
        return Err(
            AnalysisMeshValidationError::TetrahedronOptimizationQualityRegression {
                metric: "min_exact_scaled_jacobian".to_string(),
                initial_value: stable_float(initial_min),
                final_value: stable_float(final_min),
            },
        );
    }

    let initial_max = backend.tetrahedron_optimization_initial_max_aspect_ratio;
    let final_max = backend.tetrahedron_optimization_final_max_aspect_ratio;
    if initial_max.is_finite()
        && final_max.is_finite()
        && initial_max > 0.0
        && final_max > initial_max + 1.0e-12
    {
        return Err(
            AnalysisMeshValidationError::TetrahedronOptimizationQualityRegression {
                metric: "max_aspect_ratio".to_string(),
                initial_value: stable_float(initial_max),
                final_value: stable_float(final_max),
            },
        );
    }

    Ok(())
}

fn stable_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12e}")
    } else {
        value.to_string()
    }
}
