//! Authored acoustic cavity-mode and pressure-transient intent; placement stays downstream.
use runmat_analysis_core::{
    AccuracyTolerance, AnalysisProcedure, ErrorContributionKind, PhysicsFamily,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompiledAnalysisStudy;
use crate::{
    execution::{AcousticResponseSolver, AcousticResponseWorkload},
    physics::acoustic::{AcousticModalPolicy, AcousticTransientPolicy},
};

/// Cavity-mode and transient controls come from the authored step; the policy carries none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticResponseStudyPlanningPolicy {}

/// Is this a standalone acoustic step served by the cavity-mode/transient workload?
pub fn is_acoustic_response_study(study: &CompiledAnalysisStudy) -> bool {
    matches!(
        study.model.sequence.steps.as_slice(),
        [step] if step.family == PhysicsFamily::Acoustic
            && matches!(
                step.procedure,
                AnalysisProcedure::ModalExtraction | AnalysisProcedure::TransientResponse
            )
    )
}

/// The tightest authored relative algebraic target becomes the eigenpair residual bound for
/// cavity modes and the per-step effective-operator residual bound for a transient.
pub fn plan_acoustic_response_study(
    study: &CompiledAnalysisStudy,
    policy: AcousticResponseStudyPlanningPolicy,
) -> Result<AcousticResponseWorkload, AcousticResponseStudyPlanningError> {
    // The policy carries no controls today; destructuring keeps any future field explicit.
    let AcousticResponseStudyPlanningPolicy {} = policy;
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(AcousticResponseStudyPlanningError::UnsupportedSequence);
    };
    let target = step
        .accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .filter(|value| value.is_finite() && *value > 0.0 && *value < 1.0)
        .ok_or(AcousticResponseStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    let solver = match step.procedure {
        AnalysisProcedure::ModalExtraction => AcousticResponseSolver::Modal {
            policy: AcousticModalPolicy {
                relative_tolerance: target,
                // Inner shift-invert solves must be tighter than the eigen residual.
                linear_relative_tolerance: (target * 1.0e-4).max(1.0e-14),
                ..AcousticModalPolicy::default()
            },
        },
        AnalysisProcedure::TransientResponse => AcousticResponseSolver::Transient {
            policy: AcousticTransientPolicy {
                linear_relative_tolerance: target,
                ..AcousticTransientPolicy::default()
            },
        },
        _ => return Err(AcousticResponseStudyPlanningError::UnsupportedProcedure),
    };
    let workload = AcousticResponseWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        solver,
    };
    workload.validate().map_err(|error| {
        AcousticResponseStudyPlanningError::UnsupportedIntent(error.to_string())
    })?;
    Ok(workload)
}

#[derive(Debug, Error)]
pub enum AcousticResponseStudyPlanningError {
    #[error("acoustic response planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("acoustic response planning admits acoustic modal and transient procedures")]
    UnsupportedProcedure,
    #[error("acoustic response planning requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("unsupported acoustic study intent: {0}")]
    UnsupportedIntent(String),
}
