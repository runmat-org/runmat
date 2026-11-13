//! Cancellation returns no new block and leaves the original immutable history intact.
use super::*;
pub(super) fn reject_cancelled(task: &PreparedNonlinearSolidBlockTask) {
    let before = task.inputs.clone();
    let token = runmat_analysis_fea::FeaCancellationToken::new();
    token.cancel();
    let guard = runmat_analysis_fea::replace_fea_progress_context(
        runmat_analysis_fea::FeaProgressContext {
            cancellation: token,
            ..runmat_analysis_fea::FeaProgressContext::default()
        },
    );
    let error = super::super::super::NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            host: &task.host,
            kind: AnalysisWorkKind::SolveNonlinearSolidBlock,
            inputs: &task.inputs,
            output_policy: task.output_policy,
        })
        .unwrap_err();
    drop(guard);
    assert!(
        error.to_string().to_lowercase().contains("cancel"),
        "{error}"
    );
    assert_eq!(task.inputs, before);
}
