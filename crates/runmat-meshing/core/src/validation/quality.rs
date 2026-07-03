use crate::{artifact::AnalysisMeshArtifact, quality::QualityThresholds};

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
    Ok(())
}
