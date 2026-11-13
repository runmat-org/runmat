//! Shared host error adaptation; durability uncertainty never becomes invalid physics.
use crate::analysis::storage::AnalysisPublicationError;
use crate::operations::{
    OperationContext, OperationErrorEnvelope, OperationErrorSeverity, OperationErrorSpec,
    OperationErrorType,
};

#[cfg(test)]
mod tests;

/// Preserve the evaluator's typed capacity cause at whole-study preparation. Other
/// evaluation failures retain their original rejection/cancellation message.
pub(super) fn assessment_failure(
    error: crate::analysis::assessment::AssessmentEvaluationError,
) -> AnalysisPublicationError {
    match error {
        limited @ crate::analysis::assessment::AssessmentEvaluationError::ResourceLimit => {
            AnalysisPublicationError::ResourceLimit {
                message: limited.to_string(),
            }
        }
        rejected => AnalysisPublicationError::Rejected {
            message: rejected.to_string(),
        },
    }
}

pub(in crate::analysis) fn publication_failure(
    error: AnalysisPublicationError,
    context: &OperationContext,
    rejected: impl FnOnce(String) -> OperationErrorEnvelope,
) -> OperationErrorEnvelope {
    publication_failure_at(
        error,
        crate::analysis::ANALYSIS_RUN_STUDY_OPERATION,
        crate::analysis::ANALYSIS_RUN_STUDY_OP_VERSION,
        "RM.FEA.RUN_STUDY.PUBLICATION_RESOURCE_LIMIT",
        "RM.FEA.RUN_STUDY.PUBLICATION_DURABILITY_INDETERMINATE",
        context,
        rejected,
    )
}

pub(in crate::analysis) fn publication_failure_at(
    error: AnalysisPublicationError,
    operation: &str,
    version: &str,
    resource_code: &'static str,
    durability_code: &'static str,
    context: &OperationContext,
    rejected: impl FnOnce(String) -> OperationErrorEnvelope,
) -> OperationErrorEnvelope {
    match error {
        AnalysisPublicationError::Rejected { message } => rejected(message),
        limited @ AnalysisPublicationError::ResourceLimit { .. } => {
            crate::operations::operation_error(
                operation, version,
                context, OperationErrorSpec {
                    error_code: resource_code, error_type: OperationErrorType::Capacity,
                    retryable: false, severity: OperationErrorSeverity::Error,
                }, limited.to_string(), std::collections::BTreeMap::new(),
            )
        }
        uncertain @ (AnalysisPublicationError::DurabilityIndeterminate { .. } | AnalysisPublicationError::EvidenceDurabilityIndeterminate { .. }) => {
            crate::operations::operation_error(
                operation,
                version,
                context,
                OperationErrorSpec {
                    error_code: durability_code,
                    error_type: OperationErrorType::Backend,
                    retryable: true,
                    severity: OperationErrorSeverity::Error,
                },
                "Publication durability is not established; retain the original evidence and recover before reading results.",
                std::collections::BTreeMap::from([("publication_outcome".into(), uncertain.to_string())]),
            )
        }
    }
}
