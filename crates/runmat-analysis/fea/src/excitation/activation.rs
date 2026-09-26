use std::collections::HashSet;

use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationCaseActivation, ExcitationCaseId, ProcedureStep,
};

use super::realize::RealizationError;

/// Shared authored-action selection for realization and assembly-free resource inventory.
/// Signal values cannot change this inventory within one procedure step.
pub(crate) fn active_actions<'a>(
    model: &'a AnalysisModel,
    step: Option<&'a ProcedureStep>,
) -> Result<
    impl Iterator<Item = (&'a ExcitationAction, Vec<ExcitationCaseId>)> + 'a,
    RealizationError,
> {
    let step_id = step.map(|step| step.step_id.as_str());
    let active_cases = active_case_ids(model, step_id)?;
    Ok(model.excitations.iter().filter_map(move |action| {
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
            return None;
        }
        Some((action, case_ids))
    }))
}

pub(super) fn active_case_ids<'a>(
    model: &'a AnalysisModel,
    step_id: Option<&str>,
) -> Result<HashSet<&'a str>, RealizationError> {
    let mut active = HashSet::new();
    for case in &model.excitation_cases {
        match &case.activation {
            ExcitationCaseActivation::Always => {
                active.insert(case.case_id.0.as_str());
            }
            ExcitationCaseActivation::Steps(steps) => {
                if step_id.is_some_and(|step| steps.iter().any(|candidate| candidate == step)) {
                    active.insert(case.case_id.0.as_str());
                }
            }
            ExcitationCaseActivation::MutuallyExclusiveGroup { group_id } => {
                return Err(RealizationError::new(format!(
                    "excitation case {} belongs to mutually exclusive group {} but no case selection was supplied",
                    case.case_id.0, group_id
                )));
            }
        }
    }
    Ok(active)
}
