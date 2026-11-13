//! Explicit standalone publication; study execution consumes the same pure admission owners.
use super::*;

pub(super) fn persist_fea_run_result_with_progress(
    operation: &str,
    op_version: &str,
    artifact_error_code: &'static str,
    context: &OperationContext,
    result: &AnalysisRunResult,
) -> Result<(), OperationErrorEnvelope> {
    runmat_analysis_fea::emit_fea_progress_phase(
        operation,
        FeaProgressPhase::ArtifactPersistence,
        FeaProgressStatus::Started,
        "persisting FEA run artifact",
        None,
        None,
    );
    match storage::persist_run_result(result) {
        Ok(_record) => {
            runmat_analysis_fea::emit_fea_progress_phase(
                operation,
                FeaProgressPhase::ArtifactPersistence,
                FeaProgressStatus::Completed,
                "FEA run artifact persisted",
                None,
                None,
            );
            Ok(())
        }
        Err(err) => {
            let message = format!("failed to persist FEA run artifact: {err}");
            runmat_analysis_fea::emit_fea_progress_phase(
                operation,
                FeaProgressPhase::ArtifactPersistence,
                FeaProgressStatus::Failed,
                &message,
                None,
                None,
            );
            Err(operation_error(
                operation,
                op_version,
                context,
                OperationErrorSpec {
                    error_code: artifact_error_code,
                    error_type: OperationErrorType::Internal,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                message,
                BTreeMap::from([("run_id".to_string(), result.run_id.clone())]),
            ))
        }
    }
}

pub(super) fn publish_standalone_run(
    run: OperationEnvelope<AnalysisRunResult>,
    error_code: &'static str,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    let context = OperationContext::new(run.trace_id.clone(), run.request_id.clone());
    persist_fea_run_result_with_progress(
        &run.operation,
        &run.op_version,
        error_code,
        &context,
        &run.data,
    )?;
    Ok(run)
}

pub fn analysis_run_modal_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisModalRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_modal_with_options(model, backend, options, context)?,
        "RM.FEA.RUN_MODAL.ARTIFACT_STORE_FAILED",
    )
}

pub fn analysis_run_acoustic_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisAcousticRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_acoustic_with_options(model, backend, options, context)?,
        "RM.FEA.RUN_ACOUSTIC.ARTIFACT_STORE_FAILED",
    )
}

pub fn analysis_run_transient_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisTransientRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_transient_with_options(model, backend, options, context)?,
        "RM.FEA.RUN_TRANSIENT.ARTIFACT_STORE_FAILED",
    )
}

pub fn analysis_run_nonlinear_with_options_op(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: AnalysisNonlinearRunOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisRunResult>, OperationErrorEnvelope> {
    publish_standalone_run(
        execute_nonlinear_with_options(model, backend, options, context)?,
        "RM.FEA.RUN_NONLINEAR.ARTIFACT_STORE_FAILED",
    )
}
