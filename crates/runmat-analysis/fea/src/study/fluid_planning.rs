//! Authored steady fluid intent; no placement, topology substitution or coefficient defaults.
use super::CompiledAnalysisStudy;
use crate::{
    execution::FluidSteadyWorkload, physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_analysis_core::{
    AccuracyTarget, AccuracyTolerance, ErrorContributionKind, RequiredAccuracyEvidence,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidStudyPlanningPolicy {
    pub numerical: CanonicalFluidPolicy,
    pub geometry: FiniteVolumeProjectionPolicy,
}

/// Preserve all authored authority and tighten, never weaken, numerical stopping controls.
/// Unsupported observations/lifecycle are rejected by workload admission, not removed.
pub fn plan_fluid_study(
    study: &CompiledAnalysisStudy,
    policy: FluidStudyPlanningPolicy,
) -> Result<FluidSteadyWorkload, FluidStudyPlanningError> {
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(FluidStudyPlanningError::UnsupportedSequence);
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
        .ok_or(FluidStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    if !target.is_finite() || !(0.0..1.0).contains(&target) || target == 0.0 {
        return Err(FluidStudyPlanningError::InvalidRelativeAlgebraicTarget);
    }
    let mut numerical = policy.numerical;
    // Validate caller controls before min(): f64::min would otherwise erase a NaN.
    let workload = FluidSteadyWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        policy: numerical,
        geometry_policy: policy.geometry,
    };
    workload
        .validate()
        .map_err(|error| FluidStudyPlanningError::UnsupportedIntent(error.to_string()))?;
    numerical.linear_relative_tolerance = numerical.linear_relative_tolerance.min(target);
    // Fluid receipts certify original momentum/flux residuals and conservation, not
    // the inner Krylov residual. Honor matching authored closure targets before solving.
    // Spatial, field, absolute and unrelated evidence cannot set this dimensionless control.
    if let Some(physical) = step
        .accuracy
        .criteria
        .iter()
        .filter(|c| {
            c.target == AccuracyTarget::GlobalEquilibrium
                && c.contribution == ErrorContributionKind::Algebraic
                && matches!(
                    c.evidence,
                    RequiredAccuracyEvidence::ResidualBound
                        | RequiredAccuracyEvidence::ConservationBalance
                )
        })
        .filter_map(|c| match c.tolerance {
            AccuracyTolerance::Relative(value) => Some(value),
            AccuracyTolerance::Absolute { .. } => None,
        })
        .reduce(f64::min)
    {
        numerical.physical_relative_tolerance = numerical.physical_relative_tolerance.min(physical);
    }
    Ok(FluidSteadyWorkload {
        policy: numerical,
        ..workload
    })
}

#[derive(Debug, Error)]
pub enum FluidStudyPlanningError {
    #[error("fluid planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("a fluid analysis requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error(
        "a fluid analysis's relative algebraic accuracy target must lie strictly between zero and one"
    )]
    InvalidRelativeAlgebraicTarget,
    /// The authored steady or transient flow, conjugate heat transfer or fluid–structure
    /// study failed workload admission; the message is the workload's own.
    #[error("{0}")]
    UnsupportedIntent(String),
}
