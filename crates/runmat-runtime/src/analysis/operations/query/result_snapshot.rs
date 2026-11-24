//! Stored-results operations that answer from one consistent snapshot of the
//! result store.
use super::*;

pub fn analysis_results_by_run_id_op(
    run_id: &str,
    query: AnalysisResultsQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsData>, OperationErrorEnvelope> {
    let snapshot_context = context.clone();
    with_result_snapshot(&snapshot_context, run_id, || {
        results_by_run_id(run_id, query, context)
    })
}

pub fn analysis_field_page_by_run_id_op(
    run_id: &str,
    selection: AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisFieldPageResult>, OperationErrorEnvelope> {
    let snapshot_context = context.clone();
    with_result_snapshot(&snapshot_context, run_id, || {
        field_page_by_run_id(run_id, selection, options, context)
    })
}

/// Answer a stored-results operation from one consistent snapshot of the
/// result store (see [`storage::with_result_reads`]).
fn with_result_snapshot<T>(
    context: &OperationContext,
    run_id: &str,
    operation: impl FnOnce() -> Result<T, OperationErrorEnvelope>,
) -> Result<T, OperationErrorEnvelope> {
    storage::with_result_reads(operation).map_err(|error| {
        operation_error(
            ANALYSIS_RESULTS_OPERATION,
            ANALYSIS_RESULTS_OP_VERSION,
            context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to open the saved FEA results: {error}"),
            BTreeMap::from([("run_id".to_string(), run_id.to_string())]),
        )
    })?
}
