//! Explicit authored scalar electrical intent; placement and execution remain downstream.
use runmat_analysis_core::{
    AccuracyTolerance, AnalysisConstraint, ElectricalConstraintKind, ErrorContributionKind,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompiledAnalysisStudy;
use crate::execution::ElectricalStaticWorkload;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStudyPlanningPolicy {
    /// Steady conduction at a solved temperature field (an electrothermal operating point):
    /// one temperature per canonical volume element; conductivity follows each material's
    /// temperature law. `None` uses the materials' reference conductivity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element_temperature_k: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operating_time_s: Option<f64>,
}

/// Preserves the explicit equation, model and tightest authored relative algebraic target.
/// This initial envelope is fresh singleton static/direct, with prescribed real potentials
/// or references and natural insulation. It does not admit workers or certify accuracy.
pub fn plan_electrical_study(
    study: &CompiledAnalysisStudy,
    policy: ElectricalStudyPlanningPolicy,
) -> Result<ElectricalStaticWorkload, ElectricalStudyPlanningError> {
    let [step] = study.model.sequence.steps.as_slice() else {
        return Err(ElectricalStudyPlanningError::UnsupportedSequence);
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
        .ok_or(ElectricalStudyPlanningError::MissingRelativeAlgebraicTarget)?;
    let workload = ElectricalStaticWorkload {
        model: study.model.clone(),
        step_id: step.step_id.clone(),
        relative_residual_target,
        element_temperature_k: policy.element_temperature_k.clone(),
        operating_time_s: policy.operating_time_s,
    };
    workload
        .validate()
        .map_err(|error| ElectricalStudyPlanningError::UnsupportedIntent(error.to_string()))?;
    if workload.model.constraints.iter().any(|constraint| {
        !matches!(constraint,
        AnalysisConstraint::Electrical(value) if matches!(value.kind,
            ElectricalConstraintKind::PrescribedPotential { .. } |
            ElectricalConstraintKind::PotentialReference { .. }))
    }) {
        return Err(ElectricalStudyPlanningError::UnsupportedConstraint);
    }
    Ok(workload)
}

#[derive(Debug, Error)]
pub enum ElectricalStudyPlanningError {
    #[error("electrical planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("electrical planning requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("electrical planning currently supports only prescribed potentials and potential references")]
    UnsupportedConstraint,
    #[error("unsupported electrical study intent: {0}")]
    UnsupportedIntent(String),
}
