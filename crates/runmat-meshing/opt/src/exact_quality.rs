use serde::{Deserialize, Serialize};

pub const MODULE_PURPOSE: &str =
    "exact quality repair after protected constraints and untangling are present";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronExactQualityRepairOptions {
    pub min_exact_scaled_jacobian: f64,
    pub min_exact_scaled_jacobian_improvement: f64,
    pub exact_quality_tolerance: f64,
}

impl Default for TetrahedronExactQualityRepairOptions {
    fn default() -> Self {
        Self {
            min_exact_scaled_jacobian: 0.15,
            min_exact_scaled_jacobian_improvement: 1.0e-12,
            exact_quality_tolerance: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronExactQuality {
    pub tetrahedron_id: u32,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronExactQualityRepairRejectionReason {
    EmptyPatch,
    NonFiniteQuality,
    ViolationCountNotReduced,
    MinimumExactQualityRegressed,
    MinimumExactQualityDoesNotImprove,
}

impl TetrahedronExactQualityRepairRejectionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmptyPatch => "empty_patch",
            Self::NonFiniteQuality => "non_finite_quality",
            Self::ViolationCountNotReduced => "violation_count_not_reduced",
            Self::MinimumExactQualityRegressed => "minimum_exact_quality_regressed",
            Self::MinimumExactQualityDoesNotImprove => "minimum_exact_quality_does_not_improve",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronExactQualityRepairEvaluation {
    pub accepted: bool,
    pub initial_violation_count: usize,
    pub candidate_violation_count: usize,
    pub initial_min_exact_scaled_jacobian: f64,
    pub candidate_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub rejection_reason: Option<TetrahedronExactQualityRepairRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronExactQualityRepairError {
    InvalidOptions,
}

pub fn evaluate_tetrahedron_exact_quality_repair_candidate(
    current: &[TetrahedronExactQuality],
    candidate: &[TetrahedronExactQuality],
    options: TetrahedronExactQualityRepairOptions,
) -> Result<TetrahedronExactQualityRepairEvaluation, TetrahedronExactQualityRepairError> {
    validate_options(options)?;
    let current_summary = summarize_quality(current, options);
    let candidate_summary = summarize_quality(candidate, options);
    let rejection_reason =
        exact_quality_rejection_reason(current_summary, candidate_summary, options);

    Ok(TetrahedronExactQualityRepairEvaluation {
        accepted: rejection_reason.is_none(),
        initial_violation_count: current_summary.violation_count,
        candidate_violation_count: candidate_summary.violation_count,
        initial_min_exact_scaled_jacobian: current_summary.min_exact_scaled_jacobian,
        candidate_min_exact_scaled_jacobian: candidate_summary.min_exact_scaled_jacobian,
        rejection_reason,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TetrahedronExactQualitySummary {
    count: usize,
    violation_count: usize,
    min_exact_scaled_jacobian: f64,
    finite: bool,
}

fn summarize_quality(
    tetrahedra: &[TetrahedronExactQuality],
    options: TetrahedronExactQualityRepairOptions,
) -> TetrahedronExactQualitySummary {
    let mut summary = TetrahedronExactQualitySummary {
        count: tetrahedra.len(),
        violation_count: 0,
        min_exact_scaled_jacobian: f64::INFINITY,
        finite: true,
    };
    for tetrahedron in tetrahedra {
        summary.finite &= tetrahedron.exact_scaled_jacobian.is_finite();
        summary.violation_count +=
            usize::from(tetrahedron.exact_scaled_jacobian < options.min_exact_scaled_jacobian);
        summary.min_exact_scaled_jacobian = summary
            .min_exact_scaled_jacobian
            .min(tetrahedron.exact_scaled_jacobian);
    }
    if tetrahedra.is_empty() {
        summary.min_exact_scaled_jacobian = 0.0;
    }
    summary
}

fn exact_quality_rejection_reason(
    current: TetrahedronExactQualitySummary,
    candidate: TetrahedronExactQualitySummary,
    options: TetrahedronExactQualityRepairOptions,
) -> Option<TetrahedronExactQualityRepairRejectionReason> {
    if current.count == 0 || candidate.count == 0 {
        return Some(TetrahedronExactQualityRepairRejectionReason::EmptyPatch);
    }
    if !current.finite || !candidate.finite {
        return Some(TetrahedronExactQualityRepairRejectionReason::NonFiniteQuality);
    }
    if candidate.violation_count >= current.violation_count && current.violation_count > 0 {
        return Some(TetrahedronExactQualityRepairRejectionReason::ViolationCountNotReduced);
    }
    if candidate.min_exact_scaled_jacobian + options.exact_quality_tolerance
        < current.min_exact_scaled_jacobian
    {
        return Some(TetrahedronExactQualityRepairRejectionReason::MinimumExactQualityRegressed);
    }
    if candidate.min_exact_scaled_jacobian
        <= current.min_exact_scaled_jacobian + options.min_exact_scaled_jacobian_improvement
    {
        return Some(
            TetrahedronExactQualityRepairRejectionReason::MinimumExactQualityDoesNotImprove,
        );
    }
    None
}

fn validate_options(
    options: TetrahedronExactQualityRepairOptions,
) -> Result<(), TetrahedronExactQualityRepairError> {
    if !options.min_exact_scaled_jacobian.is_finite()
        || options.min_exact_scaled_jacobian < 0.0
        || !options.min_exact_scaled_jacobian_improvement.is_finite()
        || options.min_exact_scaled_jacobian_improvement < 0.0
        || !options.exact_quality_tolerance.is_finite()
        || options.exact_quality_tolerance < 0.0
    {
        return Err(TetrahedronExactQualityRepairError::InvalidOptions);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_candidate_that_reduces_exact_quality_violations() {
        let current = [quality(1, 0.05), quality(2, 0.5)];
        let candidate = [quality(1, 0.2), quality(2, 0.5)];

        let evaluation = evaluate_tetrahedron_exact_quality_repair_candidate(
            &current,
            &candidate,
            TetrahedronExactQualityRepairOptions::default(),
        )
        .expect("exact-quality evaluation should succeed");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.initial_violation_count, 1);
        assert_eq!(evaluation.candidate_violation_count, 0);
        assert_eq!(evaluation.rejection_reason, None);
    }

    #[test]
    fn rejects_candidate_that_keeps_violation_count() {
        let current = [quality(1, 0.05)];
        let candidate = [quality(1, 0.10)];

        let evaluation = evaluate_tetrahedron_exact_quality_repair_candidate(
            &current,
            &candidate,
            TetrahedronExactQualityRepairOptions::default(),
        )
        .expect("exact-quality evaluation should succeed");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(TetrahedronExactQualityRepairRejectionReason::ViolationCountNotReduced)
        );
    }

    #[test]
    fn rejects_candidate_that_regresses_minimum_exact_quality() {
        let current = [quality(1, 0.2), quality(2, 0.3)];
        let candidate = [quality(1, 0.25), quality(2, 0.1)];

        let evaluation = evaluate_tetrahedron_exact_quality_repair_candidate(
            &current,
            &candidate,
            TetrahedronExactQualityRepairOptions::default(),
        )
        .expect("exact-quality evaluation should succeed");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(TetrahedronExactQualityRepairRejectionReason::MinimumExactQualityRegressed)
        );
    }

    #[test]
    fn rejects_non_finite_candidate_quality() {
        let evaluation = evaluate_tetrahedron_exact_quality_repair_candidate(
            &[quality(1, 0.1)],
            &[quality(1, f64::NAN)],
            TetrahedronExactQualityRepairOptions::default(),
        )
        .expect("exact-quality evaluation should succeed");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(TetrahedronExactQualityRepairRejectionReason::NonFiniteQuality)
        );
    }

    fn quality(tetrahedron_id: u32, exact_scaled_jacobian: f64) -> TetrahedronExactQuality {
        TetrahedronExactQuality {
            tetrahedron_id,
            exact_scaled_jacobian,
        }
    }
}
