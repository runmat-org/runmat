//! Explicit authored scalar electrical intent; placement and execution remain downstream.
use runmat_analysis_core::{
    AccuracyTolerance, AnalysisConstraint, ElectricalConstraintKind, ErrorContributionKind,
    FormulationRequirement,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CompiledAnalysisStudy;
use crate::execution::ElectricalStaticWorkload;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectricalStudyPlanningPolicy {
    pub formulation_requirement: FormulationRequirement,
}

/// Preserves the explicit equation, model and tightest authored relative algebraic target.
/// This initial envelope is fresh singleton static/direct, with prescribed real potentials
/// or references and natural insulation. It does not admit workers or certify accuracy.
pub fn plan_electrical_study(
    study: &CompiledAnalysisStudy,
    policy: ElectricalStudyPlanningPolicy,
) -> Result<ElectricalStaticWorkload, ElectricalStudyPlanningError> {
    if policy.formulation_requirement != study.formulation_requirement {
        return Err(ElectricalStudyPlanningError::FormulationRequirementMismatch);
    }
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
        formulation_requirement: policy.formulation_requirement,
        relative_residual_target,
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
    #[error("electrical planning requirement differs from the compiled study")]
    FormulationRequirementMismatch,
    #[error("electrical planning requires a standalone singleton step")]
    UnsupportedSequence,
    #[error("electrical planning requires an authored relative algebraic accuracy target")]
    MissingRelativeAlgebraicTarget,
    #[error("electrical planning currently supports only prescribed potentials and potential references")]
    UnsupportedConstraint,
    #[error("unsupported electrical study intent: {0}")]
    UnsupportedIntent(String),
}
