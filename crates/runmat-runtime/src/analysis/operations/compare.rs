use super::super::*;
use super::run_solve_ms;

pub fn analysis_results_compare_op(
    query: AnalysisResultsCompareQuery,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisResultsCompareData>, OperationErrorEnvelope> {
    let baseline = storage::load_run_result(&query.baseline_run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load baseline FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.baseline_run_id.clone())]),
        )
    })?;
    let Some(baseline) = baseline else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "FEA baseline run_id '{}' was not found",
                query.baseline_run_id
            ),
            BTreeMap::from([("run_id".to_string(), query.baseline_run_id.clone())]),
        ));
    };

    let candidate = storage::load_run_result(&query.candidate_run_id).map_err(|err| {
        operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.ARTIFACT_STORE_FAILED",
                error_type: OperationErrorType::Internal,
                retryable: true,
                severity: OperationErrorSeverity::Error,
            },
            format!("failed to load candidate FEA run artifact: {err}"),
            BTreeMap::from([("run_id".to_string(), query.candidate_run_id.clone())]),
        )
    })?;
    let Some(candidate) = candidate else {
        return Err(operation_error(
            ANALYSIS_RESULTS_COMPARE_OPERATION,
            ANALYSIS_RESULTS_COMPARE_OP_VERSION,
            &context,
            OperationErrorSpec {
                error_code: "RM.FEA.RESULTS_COMPARE.RUN_NOT_FOUND",
                error_type: OperationErrorType::Input,
                retryable: false,
                severity: OperationErrorSeverity::Error,
            },
            format!(
                "FEA candidate run_id '{}' was not found",
                query.candidate_run_id
            ),
            BTreeMap::from([("run_id".to_string(), query.candidate_run_id.clone())]),
        ));
    };

    let baseline_solve_ms = run_solve_ms(&baseline);
    let candidate_solve_ms = run_solve_ms(&candidate);
    let failed_increment_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(b.failed_increments as i64 - a.failed_increments as i64),
        _ => None,
    };
    let max_iteration_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(
            b.iteration_counts.iter().copied().max().unwrap_or(0) as i64
                - a.iteration_counts.iter().copied().max().unwrap_or(0) as i64,
        ),
        _ => None,
    };
    let nonlinear_spike_count_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => Some(b.iteration_spike_count as i64 - a.iteration_spike_count as i64),
        _ => None,
    };
    let nonlinear_stall_count_delta = match (
        baseline.nonlinear_results.as_ref(),
        candidate.nonlinear_results.as_ref(),
    ) {
        (Some(a), Some(b)) => {
            Some(b.convergence_stall_count as i64 - a.convergence_stall_count as i64)
        }
        _ => None,
    };

    let data = AnalysisResultsCompareData {
        baseline_run_id: baseline.run_id,
        candidate_run_id: candidate.run_id,
        numerically_acceptable_changed: baseline.numerically_acceptable
            != candidate.numerically_acceptable,
        run_status_changed: baseline.run_status != candidate.run_status,
        quality_reason_count_delta: candidate.quality_reasons.len() as i64
            - baseline.quality_reasons.len() as i64,
        failed_increment_delta,
        max_iteration_delta,
        nonlinear_spike_count_delta,
        nonlinear_stall_count_delta,
        solve_ms_delta: match (baseline_solve_ms, candidate_solve_ms) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        },
    };

    Ok(OperationEnvelope::new(
        ANALYSIS_RESULTS_COMPARE_OPERATION,
        ANALYSIS_RESULTS_COMPARE_OP_VERSION,
        &context,
        data,
    ))
}
