//! Typed independent-domain history selection; no overloaded field-ID naming convention.
use super::super::*;
use runmat_analysis_core::ObservationId;
use runmat_analysis_fea::results::{
    field::RecoveredFieldSample, observation::ObservationHistorySample,
};

pub fn analysis_observation_history_sample_by_run_id_op(
    run_id: &str,
    observation_id: ObservationId,
    sample: RecoveredFieldSample,
    context: OperationContext,
) -> Result<OperationEnvelope<ObservationHistorySample>, OperationErrorEnvelope> {
    let lookup = storage::load_observation_history_sample(run_id, &observation_id, sample);
    let (code, kind, message) = match lookup {
        Ok(storage::ObservationHistoryLookup::Sample(sample)) => {
            return Ok(OperationEnvelope::new(
                ANALYSIS_RESULTS_OPERATION,
                ANALYSIS_RESULTS_OP_VERSION,
                &context,
                *sample,
            ))
        }
        Ok(storage::ObservationHistoryLookup::RunNotFound) => (
            "RM.FEA.RESULTS.RUN_NOT_FOUND",
            OperationErrorType::Input,
            "requested run was not found".into(),
        ),
        Ok(storage::ObservationHistoryLookup::ObservationNotFound) => (
            "RM.FEA.RESULTS.OBSERVATION_NOT_FOUND",
            OperationErrorType::Input,
            "requested history was not produced by this run".into(),
        ),
        Ok(storage::ObservationHistoryLookup::SampleNotFound) => (
            "RM.FEA.RESULTS.SAMPLE_NOT_FOUND",
            OperationErrorType::Input,
            "requested history sample was not retained".into(),
        ),
        Err(error) => (
            "RM.FEA.RESULTS.OBSERVATION_ARTIFACT_INVALID",
            OperationErrorType::Contract,
            error,
        ),
    };
    Err(operation_error(
        ANALYSIS_RESULTS_OPERATION,
        ANALYSIS_RESULTS_OP_VERSION,
        &context,
        OperationErrorSpec {
            error_code: code,
            error_type: kind,
            retryable: false,
            severity: OperationErrorSeverity::Error,
        },
        message,
        BTreeMap::from([
            ("run_id".into(), run_id.into()),
            ("observation_id".into(), observation_id.0),
        ]),
    ))
}
