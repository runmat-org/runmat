//! Shared run persistence keeps the admitted physics operation, never a static proxy label.
use super::*;

pub(super) fn stage(
    context: &OperationContext,
    staging: &storage::AnalysisPublicationBuilder,
    run: &AnalysisRunResult,
) -> Result<(), OperationErrorEnvelope> {
    staging.stage_run(run).map(|_| ()).map_err(|error| {
        completion_storage_error(context, "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED", error)
    })
}
