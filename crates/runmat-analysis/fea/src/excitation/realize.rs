use std::{collections::HashMap, fmt};

use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use super::{
    action::realize_kind,
    activation::active_case_ids,
    model::{RealizedExcitation, RealizedExcitationPolicy, RealizedExcitations},
    sample::ExcitationSamplePoint,
    target::realize_target,
};

pub(crate) fn realize_step_excitations(
    model: &AnalysisModel,
    step: &ProcedureStep,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitations, RealizationError> {
    realize_excitations_for_step(model, Some(step), sample)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RealizationError {
    message: String,
}

impl RealizationError {
    pub(super) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RealizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for RealizationError {}

#[cfg(test)]
pub(crate) fn realize_excitations(
    model: &AnalysisModel,
    step_id: Option<&str>,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitations, RealizationError> {
    let step = step_id.and_then(|id| {
        model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id.as_str() == id)
    });
    realize_excitations_for_step(model, step, sample)
}

fn realize_excitations_for_step(
    model: &AnalysisModel,
    step: Option<&ProcedureStep>,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitations, RealizationError> {
    let step_id = step.map(|step| step.step_id.as_str());
    let signals = model
        .signals
        .iter()
        .map(|signal| (signal.signal_id().0.as_str(), signal))
        .collect::<HashMap<_, _>>();
    let active_cases = active_case_ids(model, step_id)?;
    let mut entries = Vec::new();
    for action in &model.excitations {
        let excitation = action.excitation();
        let case_ids = model
            .excitation_cases
            .iter()
            .filter(|case| {
                active_cases.contains(case.case_id.0.as_str())
                    && case.action_ids.contains(&excitation.excitation_id)
            })
            .map(|case| case.case_id.clone())
            .collect::<Vec<_>>();
        if case_ids.is_empty()
            || step.is_some_and(|step| {
                !step.object_is_active(&runmat_analysis_core::StepObjectRef::Excitation(
                    excitation.excitation_id.0.clone(),
                ))
            })
            || (!excitation.active_step_ids.is_empty()
                && step_id.is_none_or(|step| {
                    !excitation
                        .active_step_ids
                        .iter()
                        .any(|active| active == step)
                }))
        {
            continue;
        }
        let signal = signals
            .get(excitation.signal_id.0.as_str())
            .ok_or_else(|| {
                RealizationError::new(format!(
                    "excitation {} references missing signal {}",
                    excitation.excitation_id.0, excitation.signal_id.0
                ))
            })?;
        entries.push(RealizedExcitation {
            excitation_id: excitation.excitation_id.clone(),
            case_ids,
            signal_id: excitation.signal_id.clone(),
            quantity: action.expected_quantity(),
            sample: sample.into(),
            target: realize_target(&excitation.target),
            kind: realize_kind(action, signal, sample)?,
            policy: RealizedExcitationPolicy {
                frame: excitation.frame.clone(),
                distribution: excitation.distribution,
                kinematics: excitation.kinematics,
                projection: excitation.projection,
            },
        });
    }
    Ok(RealizedExcitations::new(entries))
}
