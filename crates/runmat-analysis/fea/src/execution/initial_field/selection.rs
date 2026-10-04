//! Exact active source selection shared by consuming physics families.
use super::*;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};

pub(crate) fn validate_declarations(
    model: &AnalysisModel,
    step: &ProcedureStep,
    fields: &[InitialFieldBinding],
) -> AnalysisExecutionResult<()> {
    for (index, binding) in fields.iter().enumerate() {
        binding.validate()?;
        if fields[..index]
            .iter()
            .any(|other| other.reference() == binding.reference())
        {
            return Err(invalid("duplicate selected binding"));
        }
        if !model.initial_conditions.iter().any(|condition| {
            condition.applies_to_step(step)
                && condition.field_artifact_ref() == Some(binding.reference())
        }) {
            return Err(invalid("binding is not selected by the consuming step"));
        }
    }
    for condition in model
        .initial_conditions
        .iter()
        .filter(|condition| condition.applies_to_step(step))
    {
        if let Some(reference) = condition.field_artifact_ref() {
            if !fields.iter().any(|field| field.reference() == reference) {
                return Err(invalid("selected condition has no closed field binding"));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_selection(
    model: &AnalysisModel,
    step: &ProcedureStep,
    topology: Digest,
    node_count: usize,
    fields: &[InitialFieldBinding],
) -> AnalysisExecutionResult<()> {
    validate_declarations(model, step, fields)?;
    for condition in model
        .initial_conditions
        .iter()
        .filter(|condition| condition.applies_to_step(step))
    {
        if let Some(reference) = condition.field_artifact_ref() {
            let binding = fields
                .iter()
                .find(|field| field.reference() == reference)
                .ok_or_else(|| invalid("selected condition has no closed field binding"))?;
            binding.admit(condition, topology, node_count)?;
        }
    }
    Ok(())
}
