//! Cancellation at the real prepared boundary must preserve private original children.
use super::*;

pub(super) fn reject_and_retry(
    prepared: runmat_runtime::analysis::PreparedStudyPublication,
    result: &StructuralStudyResult,
    identity: AnalysisStudyCompletionIdentity,
    store: &FilesystemAnalysisArtifactStore,
) -> runmat_runtime::analysis::PreparedStudyPublication {
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    {
        let _guard = runmat_runtime::interrupt::replace_interrupt(Some(cancellation.token()));
        assert!(prepared.publish_without_function_execution().is_err());
        assert!(
            store.list_run_metadata().unwrap().is_empty(),
            "cancelled original study must expose no children"
        );
    }
    // Restore only the interrupt scope. Identity, artifacts, host allowance and original
    // numerical results are unchanged, with no direct repair of retained objects.
    runmat_runtime::analysis::prepare_structural_study_publication(
        result,
        OperationContext::new(None, None),
        identity,
    )
    .unwrap()
}
