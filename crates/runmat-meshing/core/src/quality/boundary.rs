use serde::{Deserialize, Serialize};

use crate::quality::ElementQuality;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateOptions {
    pub min_exact_scaled_jacobian: f64,
    pub exact_quality_tolerance: f64,
    #[serde(default = "default_require_exact_quality_improvement")]
    pub require_exact_quality_improvement: bool,
    #[serde(default)]
    pub allow_min_exact_quality_regression_above_threshold: bool,
}

impl Default for BoundaryQualityCandidateOptions {
    fn default() -> Self {
        Self {
            min_exact_scaled_jacobian: 0.15,
            exact_quality_tolerance: 1.0e-12,
            require_exact_quality_improvement: default_require_exact_quality_improvement(),
            allow_min_exact_quality_regression_above_threshold: false,
        }
    }
}

fn default_require_exact_quality_improvement() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateConstraints {
    pub boundary_recovery_preserved: bool,
    pub target_volume_preserved: bool,
    pub source_provenance_preserved: bool,
}

impl BoundaryQualityCandidateConstraints {
    pub fn preserved() -> Self {
        Self {
            boundary_recovery_preserved: true,
            target_volume_preserved: true,
            source_provenance_preserved: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryQualityCandidateRejectionReason {
    BoundaryRecoveryRegressed,
    TargetVolumeRegressed,
    SourceProvenanceRegressed,
    ExactQualityViolationRegressed,
    MinimumExactQualityRegressed,
    ExactQualityNotImproved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateEvaluation {
    pub accepted: bool,
    pub initial_exact_quality_violation_count: usize,
    pub final_exact_quality_violation_count: usize,
    pub initial_min_exact_scaled_jacobian: f64,
    pub final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub rejection_reason: Option<BoundaryQualityCandidateRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryQualityCandidateError {
    InvalidOptions,
    NonFiniteQuality { element_id: String },
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_boundary_candidate_that_reduces_exact_quality_violations() {
        let current = vec![quality("a", 0.10), quality("b", 0.30)];
        let proposed = vec![quality("a", 0.20), quality("b", 0.30)];

        let evaluation = evaluate_boundary_quality_candidate(
            &current,
            &proposed,
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect("candidate should evaluate");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.initial_exact_quality_violation_count, 1);
        assert_eq!(evaluation.final_exact_quality_violation_count, 0);
        assert_eq!(evaluation.rejection_reason, None);
    }

    #[test]
    fn accepts_boundary_candidate_that_improves_minimum_quality() {
        let current = vec![quality("a", 0.18), quality("b", 0.30)];
        let proposed = vec![quality("a", 0.24), quality("b", 0.30)];

        let evaluation = evaluate_boundary_quality_candidate(
            &current,
            &proposed,
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect("candidate should evaluate");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.initial_exact_quality_violation_count, 0);
        assert_eq!(evaluation.final_exact_quality_violation_count, 0);
        assert_eq!(evaluation.initial_min_exact_scaled_jacobian, 0.18);
        assert_eq!(evaluation.final_min_exact_scaled_jacobian, 0.24);
    }

    #[test]
    fn rejects_candidate_when_boundary_recovery_regresses() {
        let mut constraints = BoundaryQualityCandidateConstraints::preserved();
        constraints.boundary_recovery_preserved = false;

        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.10)],
            &[quality("a", 0.30)],
            constraints,
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::BoundaryRecoveryRegressed)
        );
    }

    #[test]
    fn rejects_candidate_when_target_volume_regresses() {
        let mut constraints = BoundaryQualityCandidateConstraints::preserved();
        constraints.target_volume_preserved = false;

        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.10)],
            &[quality("a", 0.30)],
            constraints,
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::TargetVolumeRegressed)
        );
    }

    #[test]
    fn rejects_candidate_when_source_provenance_regresses() {
        let mut constraints = BoundaryQualityCandidateConstraints::preserved();
        constraints.source_provenance_preserved = false;

        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.10)],
            &[quality("a", 0.30)],
            constraints,
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::SourceProvenanceRegressed)
        );
    }

    #[test]
    fn rejects_candidate_when_exact_quality_violations_regress() {
        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.20), quality("b", 0.30)],
            &[quality("a", 0.20), quality("b", 0.05)],
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::ExactQualityViolationRegressed)
        );
    }

    #[test]
    fn rejects_candidate_when_minimum_quality_regresses() {
        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.20), quality("b", 0.30)],
            &[quality("a", 0.19), quality("b", 0.30)],
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::MinimumExactQualityRegressed)
        );
    }

    #[test]
    fn accepts_above_threshold_minimum_regression_when_explicitly_allowed() {
        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.20), quality("b", 0.30)],
            &[quality("a", 0.19), quality("b", 0.30)],
            BoundaryQualityCandidateConstraints::preserved(),
            BoundaryQualityCandidateOptions {
                allow_min_exact_quality_regression_above_threshold: true,
                require_exact_quality_improvement: false,
                ..options()
            },
        )
        .expect("candidate should evaluate");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.rejection_reason, None);
    }

    #[test]
    fn rejects_candidate_when_quality_does_not_improve() {
        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.20), quality("b", 0.30)],
            &[quality("a", 0.20), quality("b", 0.30)],
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect("candidate should evaluate");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(BoundaryQualityCandidateRejectionReason::ExactQualityNotImproved)
        );
    }

    #[test]
    fn accepts_no_regression_candidate_when_improvement_is_not_required() {
        let mut options = options();
        options.require_exact_quality_improvement = false;

        let evaluation = evaluate_boundary_quality_candidate(
            &[quality("a", 0.20), quality("b", 0.30)],
            &[quality("a", 0.20), quality("b", 0.30)],
            BoundaryQualityCandidateConstraints::preserved(),
            options,
        )
        .expect("candidate should evaluate");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.rejection_reason, None);
    }

    #[test]
    fn rejects_non_finite_quality_inputs() {
        let err = evaluate_boundary_quality_candidate(
            &[quality("a", f64::NAN)],
            &[quality("a", 0.30)],
            BoundaryQualityCandidateConstraints::preserved(),
            options(),
        )
        .expect_err("non-finite exact quality should fail");

        assert_eq!(
            err,
            BoundaryQualityCandidateError::NonFiniteQuality {
                element_id: "a".to_string()
            }
        );
    }

    fn quality(element_id: &str, exact_scaled_jacobian: f64) -> ElementQuality {
        ElementQuality {
            element_id: element_id.to_string(),
            scaled_jacobian: exact_scaled_jacobian.max(0.0),
            exact_scaled_jacobian,
            aspect_ratio: 1.0,
            volume_m3: 1.0,
        }
    }

    fn options() -> BoundaryQualityCandidateOptions {
        BoundaryQualityCandidateOptions {
            min_exact_scaled_jacobian: 0.15,
            exact_quality_tolerance: 1.0e-12,
            require_exact_quality_improvement: true,
            allow_min_exact_quality_regression_above_threshold: false,
        }
    }
}
