//! Authored magnetic intent remains magnetic and independent of placement.
use super::CompiledAnalysisStudy;
use crate::execution::MagnetostaticWorkload;
use runmat_analysis_core::{AccuracyTolerance, ErrorContributionKind, FormulationRequirement};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MagneticStudyPlanningPolicy {
    pub formulation_requirement: FormulationRequirement,
}

/// Preserve the model and the tightest authored relative algebraic criterion.
/// Canonical magnetic validation rejects unsupported lifecycle and coupling;
/// this boundary never substitutes a scalar electrical equation or infers a gauge.
pub fn plan_magnetic_study(
    study: &CompiledAnalysisStudy,
    policy: MagneticStudyPlanningPolicy,
) -> Result<MagnetostaticWorkload, MagneticStudyPlanningError> {
    if policy.formulation_requirement != study.formulation_requirement {
        return Err(MagneticStudyPlanningError::FormulationRequirementMismatch);
    }
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(MagneticStudyPlanningError::UnsupportedSequence);
    };
    let relative_residual_target = step
        .accuracy
        .criteria
        .iter()
        .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic)
        .filter_map(|criterion| match criterion.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
        .ok_or(MagneticStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    let workload = MagnetostaticWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        formulation_requirement: policy.formulation_requirement,
        relative_residual_target,
    };
    workload
        .validate()
        .map_err(|error| MagneticStudyPlanningError::UnsupportedIntent(error.to_string()))?;
    Ok(workload)
}

#[derive(Debug, Error)]
pub enum MagneticStudyPlanningError {
    #[error("magnetic planning requirement differs from the compiled study")]
    FormulationRequirementMismatch,
    #[error("magnetic planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("magnetic planning requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("unsupported magnetic study intent: {0}")]
    UnsupportedIntent(String),
}
