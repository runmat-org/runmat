use std::collections::HashSet;

use runmat_analysis_core::{AnalysisModel, ExcitationCaseActivation};

use super::realize::RealizationError;

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
