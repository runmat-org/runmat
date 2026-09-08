use std::{collections::HashMap, fmt};

use runmat_analysis_core::{AnalysisModel, IndependentDomain};

use super::{
    action::realize_kind,
    activation::active_case_ids,
    model::{RealizedExcitation, RealizedExcitationPolicy, RealizedExcitations},
    sample::ExcitationSamplePoint,
    target::realize_target,
};

pub(crate) fn realize_model_excitations(
    model: &AnalysisModel,
) -> Result<RealizedExcitations, RealizationError> {
    let sample = if let Some(domain) = &model.electromagnetic {
        ExcitationSamplePoint::FrequencyHertz(domain.reference_frequency.hertz)
    } else {
        match model
            .sequence
            .steps
            .first()
            .map(|step| &step.independent_domain)
        {
            Some(IndependentDomain::TimeSeconds { start, .. }) => {
                ExcitationSamplePoint::TimeSeconds(*start)
            }
            Some(IndependentDomain::StudyParameter {
                parameter_id,
                canonical_values,
            }) => ExcitationSamplePoint::StudyParameter {
                parameter_id,
                canonical_value: canonical_values.first().copied().ok_or_else(|| {
                    RealizationError::new("study-parameter procedure has no canonical values")
                })?,
            },
            _ => ExcitationSamplePoint::Static,
        }
    };
    let step_id = model
        .sequence
        .steps
        .first()
        .map(|step| step.step_id.as_str());
    realize_excitations(model, step_id, sample)
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

pub(crate) fn realize_excitations(
    model: &AnalysisModel,
    step_id: Option<&str>,
    sample: ExcitationSamplePoint<'_>,
) -> Result<RealizedExcitations, RealizationError> {
    let signals = model
        .signals
        .iter()
        .map(|signal| (signal.signal_id().0.as_str(), signal))
        .collect::<HashMap<_, _>>();
    let active_cases = active_case_ids(model, step_id)?;
    let mut entries = Vec::new();
    for action in &model.excitations {
        let excitation = action.excitation();
        if !active_cases.contains(excitation.case_id.0.as_str())
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
            case_id: excitation.case_id.clone(),
            signal_id: excitation.signal_id.clone(),
            quantity: action.expected_quantity(),
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
