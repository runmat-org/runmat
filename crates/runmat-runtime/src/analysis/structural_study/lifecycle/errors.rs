use super::*;

/// Maps a placement-owned terminal failure into the structural-study operation contract.
pub fn structural_study_execution_failure(
    intent: &PreparedStructuralStudyIntent,
    failure: crate::analysis::StructuralStudyExecutionFailure,
) -> OperationErrorEnvelope {
    let (error_code, error_type, severity, message) = match failure {
        crate::analysis::StructuralStudyExecutionFailure::Failed(message) => (
            "RM.FEA.RUN_STUDY.REMOTE_EXECUTION_FAILED",
            OperationErrorType::Backend,
            OperationErrorSeverity::Error,
            message,
        ),
        crate::analysis::StructuralStudyExecutionFailure::Cancelled => (
            "RM.FEA.RUN_STUDY.EXECUTION_CANCELLED",
            OperationErrorType::Cancelled,
            OperationErrorSeverity::Error,
            "structural study execution was cancelled".to_string(),
        ),
        crate::analysis::StructuralStudyExecutionFailure::Indeterminate(message) => (
            "RM.FEA.RUN_STUDY.REMOTE_EXECUTION_INDETERMINATE",
            OperationErrorType::Internal,
            OperationErrorSeverity::Fatal,
            message,
        ),
    };
    intent_completion_error(intent, error_code, error_type, severity, message)
}

pub(super) fn completion_error(
    prepared: &PreparedStudyCompletion,
    error_code: &'static str,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &prepared.context,
        OperationErrorSpec {
            error_code,
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        BTreeMap::from([("study_id".to_string(), prepared.study_id.clone())]),
    )
}

pub(super) fn intent_completion_error(
    intent: &PreparedStructuralStudyIntent,
    error_code: &'static str,
    error_type: OperationErrorType,
    severity: OperationErrorSeverity,
    message: impl Into<String>,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &intent.context,
        OperationErrorSpec {
            error_code,
            error_type,
            retryable: error_type == OperationErrorType::Internal,
            severity,
        },
        message,
        BTreeMap::from([("study_id".to_string(), intent.study_id.clone())]),
    )
}
