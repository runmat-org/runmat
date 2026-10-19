//! Exact active source selection shared by consuming physics families.
use super::*;
use runmat_analysis_core::{AnalysisModel, ProcedureStep};

pub(crate) fn validate_declarations(
    model: &AnalysisModel,
    step: &ProcedureStep,
    fields: &[InitialFieldBinding],
) -> AnalysisExecutionResult<()> {
    validate_unique(fields)?;
    for binding in fields {
        if !model.initial_conditions.iter().any(|condition| {
            condition.applies_to_step(step)
                && condition.field_artifact_ref() == Some(binding.reference())
        }) {
            return Err(invalid("binding is not selected by the consuming step"));
        }
    }
    require_selected_sources(model, std::iter::once(step), fields)
}

/// Closed source union for a complete sequence, before selecting per-step workloads.
pub(crate) fn validate_inventory(
    model: &AnalysisModel,
    fields: &[InitialFieldBinding],
) -> AnalysisExecutionResult<()> {
    validate_unique(fields)?;
    for binding in fields {
        if !model.sequence.steps.iter().any(|step| {
            model.initial_conditions.iter().any(|condition| {
                condition.applies_to_step(step)
                    && condition.field_artifact_ref() == Some(binding.reference())
            })
        }) {
            return Err(invalid("binding is not selected by any consuming step"));
        }
    }
    require_selected_sources(model, model.sequence.steps.iter(), fields)
}

fn validate_unique(fields: &[InitialFieldBinding]) -> AnalysisExecutionResult<()> {
    for (index, binding) in fields.iter().enumerate() {
        binding.validate()?;
        if fields[..index]
            .iter()
            .any(|other| other.reference() == binding.reference())
        {
            return Err(invalid("duplicate selected binding"));
        }
    }
    Ok(())
}

fn require_selected_sources<'a>(
    model: &AnalysisModel,
    steps: impl Iterator<Item = &'a ProcedureStep>,
    fields: &[InitialFieldBinding],
) -> AnalysisExecutionResult<()> {
    for step in steps {
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
