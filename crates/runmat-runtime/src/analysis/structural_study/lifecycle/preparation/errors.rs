use super::*;

pub(super) fn portable_study_kind_error(
    study: &crate::analysis::PreparedStudySequence,
    step: &crate::analysis::PreparedStudyStep,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &study.context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.PORTABLE_FAMILY_UNSUPPORTED",
            error_type: OperationErrorType::Validation,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        "portable structural studies require a structural-linear sequence, direct transient or harmonic response, modal extraction, or an authored modal-to-transient sequence",
        BTreeMap::from([
            ("study_id".to_string(), study.spec.study_id.clone()),
            (
                "step_id".to_string(),
                step.procedure_step.step_id.as_str().to_string(),
            ),
        ]),
    )
}

pub(super) fn portable_study_contract_error(
    study: &crate::analysis::PreparedStudySequence,
    message: String,
) -> OperationErrorEnvelope {
    operation_error(
        ANALYSIS_RUN_STUDY_OPERATION,
        ANALYSIS_RUN_STUDY_OP_VERSION,
        &study.context,
        OperationErrorSpec {
            error_code: "RM.FEA.RUN_STUDY.INVALID_PORTABLE_INTENT",
            error_type: OperationErrorType::Contract,
            retryable: false,
            severity: OperationErrorSeverity::Fatal,
        },
        message,
        BTreeMap::from([("study_id".to_string(), study.spec.study_id.clone())]),
    )
}
