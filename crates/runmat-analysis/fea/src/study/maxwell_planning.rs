//! Authored harmonic electric-field intent remains Maxwell and placement-neutral.
use super::CompiledAnalysisStudy;
use crate::execution::MaxwellHarmonicWorkload;
use runmat_analysis_core::{AccuracyTolerance, ErrorContributionKind};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
pub(super) mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaxwellStudyPlanningPolicy {}

/// Preserve the model and the tightest authored relative algebraic criterion.
/// Each workload selects one authored frequency and retains the complete model.
/// Canonical harmonic validation rejects unsupported lifecycle and coupling;
/// this boundary never substitutes a scalar electrical equation or fabricates magnetic vector potential.
pub fn plan_maxwell_study(
    study: &CompiledAnalysisStudy,
    _policy: MaxwellStudyPlanningPolicy,
) -> Result<MaxwellHarmonicWorkload, MaxwellStudyPlanningError> {
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(MaxwellStudyPlanningError::UnsupportedSequence);
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
        .ok_or(MaxwellStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    let runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies) =
        &step.independent_domain
    else {
        return Err(MaxwellStudyPlanningError::UnsupportedIntent(
            "harmonic step without frequencies".into(),
        ));
    };
    let first = *frequencies
        .first()
        .ok_or_else(|| MaxwellStudyPlanningError::UnsupportedIntent("no frequencies".into()))?;
    let workload = MaxwellHarmonicWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        relative_residual_target,
        frequency: runmat_analysis_core::Frequency::new(first),
    };
    workload
        .validate()
        .map_err(|error| MaxwellStudyPlanningError::UnsupportedIntent(error.to_string()))?;
    Ok(workload)
}

#[derive(Debug, Error)]
pub enum MaxwellStudyPlanningError {
    #[error("maxwell planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("maxwell planning requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("unsupported maxwell study intent: {0}")]
    UnsupportedIntent(String),
}
