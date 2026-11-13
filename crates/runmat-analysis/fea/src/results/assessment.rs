//! Compact, attributed scalar observation inventory for acceptance and later reassessment.
//! Numerical evaluation remains in the observation owner; this projection never reduces fields.
use super::{
    field::RecoveredFieldSample,
    observation::{
        ObservationEntity, ObservationResult, SeriesReductionPayload, SeriesReductionResult,
    },
};
use crate::FeaRunResult;
use runmat_analysis_core::ObservationId;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarObservationEvidence {
    pub observation_id: ObservationId,
    pub value: f64,
    pub source_identity: Digest,
    pub sample: Option<RecoveredFieldSample>,
    pub entity: Option<ObservationEntity>,
}

/// Project an already admitted scalar result without reducing or gathering numerical fields.
pub fn scalar_observation_evidence(
    observation: &ObservationResult,
) -> Option<ScalarObservationEvidence> {
    let values = observation.field.as_host_f64()?;
    if observation.value_shape != runmat_analysis_core::ObservationValueShape::Scalar
        || values.len() != 1
        || !values[0].is_finite()
    {
        return None;
    }
    Some(ScalarObservationEvidence {
        observation_id: observation.observation_id.clone(),
        value: values[0],
        source_identity: observation.logical_identity.digest(),
        sample: Some(observation.sample),
        entity: observation.entities.first().cloned(),
    })
}

/// One scalar per canonical result; tie attribution chooses the first published canonical location.
/// Histories must be reduced by the observation engine, not by this adapter.
pub fn scalar_observation_inventory(run: &FeaRunResult) -> Vec<ScalarObservationEvidence> {
    let mut inventory = Vec::new();
    for observation in &run.observations {
        inventory.extend(scalar_observation_evidence(observation));
    }
    for series in &run.series_reductions {
        inventory.extend(scalar_series_evidence(series));
    }
    inventory
}

/// Project an already authenticated series result; this does not evaluate its samples.
pub fn scalar_series_evidence(series: &SeriesReductionResult) -> Option<ScalarObservationEvidence> {
    let SeriesReductionPayload::Values { field, extrema } = &series.payload else {
        return None;
    };
    let values = field.as_host_f64()?;
    if values.len() != 1 || !values[0].is_finite() {
        return None;
    }
    let governing = extrema.first();
    Some(ScalarObservationEvidence {
        observation_id: series.observation_id.clone(),
        value: values[0],
        source_identity: series.identity.digest(),
        sample: governing.map(|value| value.source.coordinate),
        entity: governing.and_then(|value| value.entity.clone()),
    })
}
