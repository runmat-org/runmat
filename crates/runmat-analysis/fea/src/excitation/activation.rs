use runmat_analysis_core::{
    AnalysisModel, ExcitationAction, ExcitationCaseActivation, ExcitationCaseId, ProcedureStep,
};

use super::realize::RealizationError;

#[derive(Clone, Copy)]
pub(crate) struct ActiveAction<'a> {
    pub action: &'a ExcitationAction,
    model: &'a AnalysisModel,
    step_id: Option<&'a str>,
    selection: Option<&'a runmat_analysis_core::ExcitationCaseSelection>,
}

impl<'a> ActiveAction<'a> {
    pub(crate) fn case_ids(&self) -> impl Iterator<Item = &'a ExcitationCaseId> + Clone {
        let action = self.action.excitation();
        let step_id = self.step_id;
        let selection = self.selection;
        self.model
            .excitation_cases
            .iter()
            .filter(move |case| {
                case_is_active(&case.activation, step_id)
                    && selection.is_none_or(|selection| selection.includes(&case.case_id))
                    && case.action_ids.contains(&action.excitation_id)
            })
            .map(|case| &case.case_id)
    }
}

/// Shared borrowed selection for realization, scheduling and resource inventory.
/// Signal values cannot change this inventory within one procedure step. Only realization
/// copies case identifiers; scheduling and preflight can inspect them without owned vectors.
pub(crate) fn active_actions<'a>(
    model: &'a AnalysisModel,
    step: Option<&'a ProcedureStep>,
) -> Result<impl Iterator<Item = ActiveAction<'a>> + Clone + 'a, RealizationError> {
    for case in &model.excitation_cases {
        if let ExcitationCaseActivation::MutuallyExclusiveGroup { group_id } = &case.activation {
            return Err(RealizationError::new(format!(
                "excitation case {} belongs to mutually exclusive group {} but no case selection was supplied",
                case.case_id.0, group_id
            )));
        }
    }
    let step_id = step.map(|step| step.step_id.as_str());
    Ok(model.excitations.iter().filter_map(move |action| {
        let excitation = action.excitation();
        let selected = ActiveAction {
            action,
            model,
            step_id,
            selection: step.map(|step| &step.excitation_case_selection),
        };
        if selected.case_ids().next().is_none()
            || step.is_some_and(|step| {
                !step.activations.is_empty()
                    && !step.object_is_active(&runmat_analysis_core::StepObjectRef::Excitation(
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
        Some(selected)
    }))
}

fn case_is_active(activation: &ExcitationCaseActivation, step: Option<&str>) -> bool {
    match activation {
        ExcitationCaseActivation::Always => true,
        ExcitationCaseActivation::Steps(steps) => {
            step.is_some_and(|step| steps.iter().any(|candidate| candidate == step))
        }
        // Rejected before constructing the selection iterator.
        ExcitationCaseActivation::MutuallyExclusiveGroup { .. } => false,
    }
}
