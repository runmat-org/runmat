//! Portable modal-transient input preparation with original source bindings.
use super::*;

pub(in crate::analysis) fn prepare_structural_modal_transient_template(
    study: &super::super::super::PreparedStudySequence,
    step: &super::super::super::PreparedStudyStep,
    options: &crate::analysis::AnalysisTransientRunOptions,
) -> Result<runmat_analysis_fea::execution::StructuralModalTransientTemplate, OperationErrorEnvelope>
{
    let template = runmat_analysis_fea::execution::StructuralModalTransientTemplate {
        schema_version: 2,
        initial_fields: storage::resolve_initial_fields(
            &study.model,
            &step.procedure_step,
            step.procedure_step
                .accuracy
                .resources
                .resident_byte_budget(),
        )
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?,
        model: study.model.clone(),
        step_id: step.procedure_step.step_id.clone(),
        backend: step.authored_execution.backend,
        time_step_policy: runmat_analysis_fea::solve::transient::StructuralTimeStepPolicy::Fixed {
            interval_count: options.time_step_policy.interval_count(),
        },
        residual_target: options.residual_target,
    };
    template
        .validate()
        .map_err(|error| portable_study_contract_error(study, error.to_string()))?;
    Ok(template)
}
