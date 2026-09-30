//! Shared run persistence keeps the admitted physics operation, never a static proxy label.
use super::*;

pub(super) fn persist(
    context: &OperationContext,
    run: &AnalysisRunResult,
) -> Result<(), OperationErrorEnvelope> {
    let operation = run_operation_for_identity(&run.run.identity).ok_or_else(|| {
        completion_error(
            context,
            "RM.FEA.RUN_STUDY.RESULT_IDENTITY_UNSUPPORTED",
            "persisted run has no admitted runtime operation",
        )
    })?;
    crate::analysis::persist_fea_run_result_with_progress(
        operation.name(),
        operation.version(),
        "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
        context,
        run,
    )
}
