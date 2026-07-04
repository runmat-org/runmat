use crate::quality::ElementQuality;

use super::{
    BoundaryQualityCandidateConstraints, BoundaryQualityCandidateError,
    BoundaryQualityCandidateEvaluation, BoundaryQualityCandidateOptions,
    BoundaryQualityCandidateRejectionReason,
};

pub fn evaluate_boundary_quality_candidate(
    current: &[ElementQuality],
    proposed: &[ElementQuality],
    constraints: BoundaryQualityCandidateConstraints,
    options: BoundaryQualityCandidateOptions,
) -> Result<BoundaryQualityCandidateEvaluation, BoundaryQualityCandidateError> {
    validate_options(options)?;
    validate_quality_set(current)?;
    validate_quality_set(proposed)?;

    let current_summary = quality_summary(current, options);
    let proposed_summary = quality_summary(proposed, options);
    let rejection_reason = if !constraints.boundary_recovery_preserved {
        Some(BoundaryQualityCandidateRejectionReason::BoundaryRecoveryRegressed)
    } else if !constraints.target_volume_preserved {
        Some(BoundaryQualityCandidateRejectionReason::TargetVolumeRegressed)
    } else if !constraints.source_provenance_preserved {
        Some(BoundaryQualityCandidateRejectionReason::SourceProvenanceRegressed)
    } else if proposed_summary.exact_quality_violation_count
        > current_summary.exact_quality_violation_count
    {
        Some(BoundaryQualityCandidateRejectionReason::ExactQualityViolationRegressed)
    } else if minimum_exact_quality_regressed(current_summary, proposed_summary, options) {
        Some(BoundaryQualityCandidateRejectionReason::MinimumExactQualityRegressed)
    } else if proposed_summary.exact_quality_violation_count
        == current_summary.exact_quality_violation_count
        && proposed_summary.min_exact_scaled_jacobian
            <= current_summary.min_exact_scaled_jacobian + options.exact_quality_tolerance
        && options.require_exact_quality_improvement
    {
        Some(BoundaryQualityCandidateRejectionReason::ExactQualityNotImproved)
    } else {
        None
    };

    Ok(BoundaryQualityCandidateEvaluation {
        accepted: rejection_reason.is_none(),
        initial_exact_quality_violation_count: current_summary.exact_quality_violation_count,
        final_exact_quality_violation_count: proposed_summary.exact_quality_violation_count,
        initial_min_exact_scaled_jacobian: current_summary.min_exact_scaled_jacobian,
        final_min_exact_scaled_jacobian: proposed_summary.min_exact_scaled_jacobian,
        rejection_reason,
    })
}

fn minimum_exact_quality_regressed(
    current: QualitySummary,
    proposed: QualitySummary,
    options: BoundaryQualityCandidateOptions,
) -> bool {
    if proposed.min_exact_scaled_jacobian + options.exact_quality_tolerance
        >= current.min_exact_scaled_jacobian
    {
        return false;
    }
    if options.allow_min_exact_quality_regression_above_threshold
        && proposed.min_exact_scaled_jacobian + options.exact_quality_tolerance
            >= options.min_exact_scaled_jacobian
    {
        return false;
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QualitySummary {
    exact_quality_violation_count: usize,
    min_exact_scaled_jacobian: f64,
}

fn quality_summary(
    elements: &[ElementQuality],
    options: BoundaryQualityCandidateOptions,
) -> QualitySummary {
    let mut exact_quality_violation_count = 0_usize;
    let mut min_exact_scaled_jacobian = f64::INFINITY;
    for element in elements {
        exact_quality_violation_count +=
            usize::from(element.exact_scaled_jacobian < options.min_exact_scaled_jacobian);
        min_exact_scaled_jacobian = min_exact_scaled_jacobian.min(element.exact_scaled_jacobian);
    }
    if elements.is_empty() {
        min_exact_scaled_jacobian = 0.0;
    }
    QualitySummary {
        exact_quality_violation_count,
        min_exact_scaled_jacobian,
    }
}

fn validate_options(
    options: BoundaryQualityCandidateOptions,
) -> Result<(), BoundaryQualityCandidateError> {
    if !options.min_exact_scaled_jacobian.is_finite()
        || options.min_exact_scaled_jacobian < 0.0
        || !options.exact_quality_tolerance.is_finite()
        || options.exact_quality_tolerance < 0.0
    {
        return Err(BoundaryQualityCandidateError::InvalidOptions);
    }
    Ok(())
}

fn validate_quality_set(elements: &[ElementQuality]) -> Result<(), BoundaryQualityCandidateError> {
    for element in elements {
        if !element.exact_scaled_jacobian.is_finite() {
            return Err(BoundaryQualityCandidateError::NonFiniteQuality {
                element_id: element.element_id.clone(),
            });
        }
    }
    Ok(())
}
