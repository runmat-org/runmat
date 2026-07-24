use serde::{Deserialize, Serialize};

pub const MODULE_PURPOSE: &str = "inversion repair after protected constraints are present";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronUntanglingOptions {
    pub near_singular_scaled_jacobian: f64,
    pub min_scaled_jacobian_improvement: f64,
}

impl Default for TetrahedronUntanglingOptions {
    fn default() -> Self {
        Self {
            near_singular_scaled_jacobian: 0.05,
            min_scaled_jacobian_improvement: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronUntanglingQuality {
    pub tetrahedron_id: u32,
    pub scaled_jacobian: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronUntanglingRejectionReason {
    EmptyPatch,
    NonFiniteQuality,
    InversionCountNotReduced,
    NearSingularCountNotReduced,
    MinimumScaledJacobianRegressed,
    MinimumScaledJacobianDoesNotImprove,
}

impl TetrahedronUntanglingRejectionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EmptyPatch => "empty_patch",
            Self::NonFiniteQuality => "non_finite_quality",
            Self::InversionCountNotReduced => "inversion_count_not_reduced",
            Self::NearSingularCountNotReduced => "near_singular_count_not_reduced",
            Self::MinimumScaledJacobianRegressed => "minimum_scaled_jacobian_regressed",
            Self::MinimumScaledJacobianDoesNotImprove => "minimum_scaled_jacobian_does_not_improve",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TetrahedronUntanglingEvaluation {
    pub accepted: bool,
    pub initial_inverted_count: usize,
    pub candidate_inverted_count: usize,
    pub initial_near_singular_count: usize,
    pub candidate_near_singular_count: usize,
    pub initial_min_scaled_jacobian: f64,
    pub candidate_min_scaled_jacobian: f64,
    #[serde(default)]
    pub rejection_reason: Option<TetrahedronUntanglingRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TetrahedronUntanglingError {
    InvalidOptions,
}

pub fn evaluate_tetrahedron_untangling_candidate(
    current: &[TetrahedronUntanglingQuality],
    candidate: &[TetrahedronUntanglingQuality],
    options: TetrahedronUntanglingOptions,
) -> Result<TetrahedronUntanglingEvaluation, TetrahedronUntanglingError> {
    validate_options(options)?;
    let current_summary = summarize_quality(current, options);
    let candidate_summary = summarize_quality(candidate, options);
    let rejection_reason = untangling_rejection_reason(current_summary, candidate_summary, options);

    Ok(TetrahedronUntanglingEvaluation {
        accepted: rejection_reason.is_none(),
        initial_inverted_count: current_summary.inverted_count,
        candidate_inverted_count: candidate_summary.inverted_count,
        initial_near_singular_count: current_summary.near_singular_count,
        candidate_near_singular_count: candidate_summary.near_singular_count,
        initial_min_scaled_jacobian: current_summary.min_scaled_jacobian,
        candidate_min_scaled_jacobian: candidate_summary.min_scaled_jacobian,
        rejection_reason,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TetrahedronUntanglingQualitySummary {
    count: usize,
    inverted_count: usize,
    near_singular_count: usize,
    min_scaled_jacobian: f64,
    finite: bool,
}

fn summarize_quality(
    tetrahedra: &[TetrahedronUntanglingQuality],
    options: TetrahedronUntanglingOptions,
) -> TetrahedronUntanglingQualitySummary {
    let mut summary = TetrahedronUntanglingQualitySummary {
        count: tetrahedra.len(),
        inverted_count: 0,
        near_singular_count: 0,
        min_scaled_jacobian: f64::INFINITY,
        finite: true,
    };
    for tetrahedron in tetrahedra {
        summary.finite &= tetrahedron.scaled_jacobian.is_finite();
        summary.inverted_count += usize::from(tetrahedron.scaled_jacobian <= 0.0);
        summary.near_singular_count +=
            usize::from(tetrahedron.scaled_jacobian < options.near_singular_scaled_jacobian);
        summary.min_scaled_jacobian = summary.min_scaled_jacobian.min(tetrahedron.scaled_jacobian);
    }
    if tetrahedra.is_empty() {
        summary.min_scaled_jacobian = 0.0;
    }
    summary
}

fn untangling_rejection_reason(
    current: TetrahedronUntanglingQualitySummary,
    candidate: TetrahedronUntanglingQualitySummary,
    options: TetrahedronUntanglingOptions,
) -> Option<TetrahedronUntanglingRejectionReason> {
    if current.count == 0 || candidate.count == 0 {
        return Some(TetrahedronUntanglingRejectionReason::EmptyPatch);
    }
    if !current.finite || !candidate.finite {
        return Some(TetrahedronUntanglingRejectionReason::NonFiniteQuality);
    }
    if candidate.inverted_count >= current.inverted_count && current.inverted_count > 0 {
        return Some(TetrahedronUntanglingRejectionReason::InversionCountNotReduced);
    }
    if current.inverted_count == 0
        && candidate.near_singular_count >= current.near_singular_count
        && current.near_singular_count > 0
    {
        return Some(TetrahedronUntanglingRejectionReason::NearSingularCountNotReduced);
    }
    if candidate.min_scaled_jacobian < current.min_scaled_jacobian {
        return Some(TetrahedronUntanglingRejectionReason::MinimumScaledJacobianRegressed);
    }
    if candidate.min_scaled_jacobian
        <= current.min_scaled_jacobian + options.min_scaled_jacobian_improvement
    {
        return Some(TetrahedronUntanglingRejectionReason::MinimumScaledJacobianDoesNotImprove);
    }
    None
}

fn validate_options(
    options: TetrahedronUntanglingOptions,
) -> Result<(), TetrahedronUntanglingError> {
    if !options.near_singular_scaled_jacobian.is_finite()
        || options.near_singular_scaled_jacobian < 0.0
        || !options.min_scaled_jacobian_improvement.is_finite()
        || options.min_scaled_jacobian_improvement < 0.0
    {
        return Err(TetrahedronUntanglingError::InvalidOptions);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_candidate_that_reduces_inversions() {
        let current = [quality(1, -0.2), quality(2, 0.1)];
        let candidate = [quality(1, 0.02), quality(2, 0.1)];

        let evaluation = evaluate_tetrahedron_untangling_candidate(
            &current,
            &candidate,
            TetrahedronUntanglingOptions::default(),
        )
        .expect("untangling evaluation should succeed");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.initial_inverted_count, 1);
        assert_eq!(evaluation.candidate_inverted_count, 0);
        assert_eq!(evaluation.rejection_reason, None);
    }

    #[test]
    fn accepts_candidate_that_reduces_near_singular_elements() {
        let current = [quality(1, 0.02), quality(2, 0.1)];
        let candidate = [quality(1, 0.06), quality(2, 0.1)];

        let evaluation = evaluate_tetrahedron_untangling_candidate(
            &current,
            &candidate,
            TetrahedronUntanglingOptions::default(),
        )
        .expect("untangling evaluation should succeed");

        assert!(evaluation.accepted);
        assert_eq!(evaluation.initial_near_singular_count, 1);
        assert_eq!(evaluation.candidate_near_singular_count, 0);
    }

    #[test]
    fn rejects_candidate_that_keeps_inversion_count() {
        let current = [quality(1, -0.2)];
        let candidate = [quality(1, -0.1)];

        let evaluation = evaluate_tetrahedron_untangling_candidate(
            &current,
            &candidate,
            TetrahedronUntanglingOptions::default(),
        )
        .expect("untangling evaluation should succeed");

        assert!(!evaluation.accepted);
        assert_eq!(
            evaluation.rejection_reason,
            Some(TetrahedronUntanglingRejectionReason::InversionCountNotReduced)
        );
    }

    #[test]
    fn rejects_candidate_with_non_finite_quality() {
        let err = evaluate_tetrahedron_untangling_candidate(
            &[quality(1, 0.1)],
            &[quality(1, f64::NAN)],
            TetrahedronUntanglingOptions::default(),
        )
        .expect("validation catches non-finite inputs in evaluation");

        assert!(!err.accepted);
        assert_eq!(
            err.rejection_reason,
            Some(TetrahedronUntanglingRejectionReason::NonFiniteQuality)
        );
    }

    fn quality(tetrahedron_id: u32, scaled_jacobian: f64) -> TetrahedronUntanglingQuality {
        TetrahedronUntanglingQuality {
            tetrahedron_id,
            scaled_jacobian,
        }
    }
}
