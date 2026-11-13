//! Whole-sequence source inventory; physical admission remains with each workload.
use super::*;
use crate::execution::InitialFieldBinding;
use runmat_analysis_core::AnalysisModel;

fn invalid(message: &str) -> StructuralStudyPlanningError {
    AnalysisExecutionError::Invalid(format!("structural study initial fields: {message}")).into()
}

pub(super) fn validate_inventory(
    model: &AnalysisModel,
    fields: &[InitialFieldBinding],
) -> Result<(), StructuralStudyPlanningError> {
    crate::execution::validate_initial_field_inventory(model, fields)?;
    for step in &model.sequence.steps {
        for condition in model
            .initial_conditions
            .iter()
            .filter(|condition| condition.applies_to_step(step))
        {
            if condition.field_artifact_ref().is_none() {
                continue;
            }
            if !(step.family == PhysicsFamily::Thermal
                || (step.family == PhysicsFamily::Structural
                    && step.procedure == AnalysisProcedure::TransientResponse))
            {
                return Err(invalid(
                    "selected procedure cannot consume an original field",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn select(
    model: &AnalysisModel,
    step: &ProcedureStep,
    fields: &[InitialFieldBinding],
) -> Result<Vec<InitialFieldBinding>, StructuralStudyPlanningError> {
    let selected = fields
        .iter()
        .filter(|field| {
            model.initial_conditions.iter().any(|condition| {
                condition.applies_to_step(step)
                    && condition.field_artifact_ref() == Some(field.reference())
            })
        })
        .collect::<Vec<_>>();
    crate::planning::resource::require_borrowed_initial_field_ceiling(
        &selected,
        runmat_canonical_codec::CanonicalLimits::platform_content(65_536, 64),
        step.accuracy.resources.resident_byte_budget(),
    )
    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let selected = selected.into_iter().cloned().collect::<Vec<_>>();
    crate::execution::validate_initial_field_declarations(model, step, &selected)?;
    Ok(selected)
}
