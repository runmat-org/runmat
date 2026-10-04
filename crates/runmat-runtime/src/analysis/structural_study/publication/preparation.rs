//! Owned-output legacy completion shares the same captured storage authority.
use super::*;

pub(in crate::analysis) fn complete_study_run(
    prepared: PreparedStudyCompletion,
    completed_steps: Vec<CompletedStudyStep>,
    states: Vec<CanonicalSemanticState>,
    frozen_project_digest: Option<runmat_execution::Digest>,
) -> Result<DeferredStudyRun, OperationErrorEnvelope> {
    let preparation =
        crate::analysis::study_publication::preparation::PublicationPreparation::capture(
            prepared.authored_study_digest,
            &(&prepared.model, &states),
        )
        .map_err(|error| {
            completion_storage_error(
                &prepared.context,
                "RM.FEA.RUN_STUDY.ARTIFACT_STORE_FAILED",
                error,
            )
        })?;
    complete_study_run_prepared(
        prepared,
        completed_steps,
        states,
        frozen_project_digest,
        preparation,
    )
}
