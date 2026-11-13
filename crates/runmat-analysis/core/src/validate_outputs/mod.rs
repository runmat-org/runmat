mod monitor;
mod sampling;
pub(crate) mod spec;

pub use sampling::observation_sampling_matches_domain;

use std::collections::{HashMap, HashSet};

use crate::{
    AnalysisModel, AnalysisValidationError, ObservationRequest, ObservationSpec, ProcedureStep,
};

pub(crate) fn validate_outputs(
    model: &AnalysisModel,
    step: &ProcedureStep,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    let mut ordinary: HashMap<&str, &ObservationSpec> = HashMap::new();
    for request in &step.observations {
        let id = request.observation_id().0.trim();
        if id.is_empty() {
            return observation_error(request, "observation id must be non-empty");
        }
        if !ids.insert(id) {
            return observation_error(request, "observation id must be unique within its step");
        }
        match request {
            ObservationRequest::Field(value)
            | ObservationRequest::History(value)
            | ObservationRequest::Probe(value)
            | ObservationRequest::Reduction(value) => {
                spec::validate_spec(model, step, request, value)?;
                ordinary.insert(value.observation_id.0.as_str(), value);
            }
            ObservationRequest::Diagnostic(value) => {
                sampling::validate_sampling(step, request, &value.sampling)?;
                sampling::validate_retention(request, &value.retention)?;
            }
            ObservationRequest::Monitor(_) => {}
        }
    }
    monitor::validate_monitors(step, &ordinary)
}

pub fn validate_observation_sampling(
    step: &ProcedureStep,
    request: &ObservationRequest,
) -> Result<(), AnalysisValidationError> {
    match request {
        ObservationRequest::Field(spec)
        | ObservationRequest::History(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec) => {
            sampling::validate_sampling(step, request, &spec.sampling)
        }
        ObservationRequest::Diagnostic(spec) => {
            sampling::validate_sampling(step, request, &spec.sampling)
        }
        ObservationRequest::Monitor(_) => Ok(()),
    }
}

/// Validate the solver-independent portion of one authored observation request.
///
/// Exact region/entity existence remains a realized-model responsibility. Quantity, shape,
/// representation, recovery, sampling, reduction, storage, and purpose contracts are complete
/// before geometry realization and must fail at authoring time.
pub fn validate_authored_observation_request(
    step: &ProcedureStep,
    request: &ObservationRequest,
) -> Result<(), AnalysisValidationError> {
    match request {
        ObservationRequest::Field(spec)
        | ObservationRequest::History(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec) => spec::validate_authored_spec(step, request, spec),
        ObservationRequest::Diagnostic(spec) => {
            sampling::validate_sampling(step, request, &spec.sampling)?;
            sampling::validate_retention(request, &spec.retention)
        }
        ObservationRequest::Monitor(_) => Ok(()),
    }
}

pub(super) fn observation_error<T>(
    request: &ObservationRequest,
    detail: impl Into<String>,
) -> Result<T, AnalysisValidationError> {
    Err(invalid_observation(request, detail))
}

pub(super) fn invalid_observation(
    request: &ObservationRequest,
    detail: impl Into<String>,
) -> AnalysisValidationError {
    AnalysisValidationError::InvalidObservation {
        observation_id: request.observation_id().0.clone(),
        detail: detail.into(),
    }
}

pub(super) fn monitor_error<T>(
    monitor_id: &str,
    detail: impl Into<String>,
) -> Result<T, AnalysisValidationError> {
    Err(AnalysisValidationError::InvalidMonitor {
        monitor_id: monitor_id.to_string(),
        detail: detail.into(),
    })
}
