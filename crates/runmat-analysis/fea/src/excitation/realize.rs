use std::fmt;

use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use super::{
    action::realize_kind,
    activation::active_actions,
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
    let mut signals = Vec::new();
    signals
        .try_reserve_exact(model.signals.len())
        .map_err(|error| RealizationError::new(error.to_string()))?;
    signals.extend(model.signals.iter());
    signals.sort_unstable_by(|left, right| left.signal_id().cmp(right.signal_id()));
    if signals
        .windows(2)
        .any(|pair| pair[0].signal_id() == pair[1].signal_id())
    {
        return Err(RealizationError::new(
            "duplicate signal identifiers in realization",
        ));
    }
    let active = active_actions(model, step)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(active.clone().count())
        .map_err(|error| RealizationError::new(error.to_string()))?;
    for selected in active {
        let action = selected.action;
        let mut case_ids = Vec::new();
        case_ids
            .try_reserve_exact(selected.case_ids().count())
            .map_err(|error| RealizationError::new(error.to_string()))?;
        case_ids.extend(selected.case_ids().cloned());
        let excitation = action.excitation();
        let signal = signals
            .binary_search_by(|signal| signal.signal_id().cmp(&excitation.signal_id))
            .map(|index| signals[index])
            .map_err(|_| {
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
