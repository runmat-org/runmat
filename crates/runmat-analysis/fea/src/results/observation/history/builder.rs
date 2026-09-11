use std::collections::VecDeque;

use runmat_analysis_core::{
    IndependentDomain, ObservationId, ObservationValueShape, RetentionPolicy,
};
use runmat_execution::Digest;

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{
    ObservationError, ObservationResult, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity,
};
use super::contract::{
    derive_history_identity, ObservationHistory, ObservationHistoryEntry,
    ObservationHistoryIdentity, ObservationHistoryManifest, ObservationHistorySample,
};
use super::schedule::{
    coordinate_follows, exact_scheduled_sample_count, sample_matches_domain, sample_matches_policy,
    validate_history_policy,
};

pub struct BoundedObservationHistory {
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    independent_domain: IndependentDomain,
    observation: ResolvedObservation,
    maximum_produced_samples: usize,
    maximum_retained_samples: usize,
    observed_sample_count: u64,
    samples: VecDeque<ObservationHistorySample>,
}

impl BoundedObservationHistory {
    pub fn new(
        plan: &ResolvedObservationPlan,
        observation_id: &ObservationId,
    ) -> Result<Self, ObservationError> {
        plan.validate_identity()?;
        let observation = plan
            .observations
            .iter()
            .find(|observation| &observation.spec.observation_id == observation_id)
            .ok_or_else(|| {
                ObservationError::Invalid(
                    "history observation is absent from its resolved plan".into(),
                )
            })?
            .clone();
        validate_history_policy(
            &observation.spec.sampling,
            &observation.spec.storage.retention,
        )?;
        let maximum_produced_samples = observation.sample_bounds.maximum_produced_samples;
        let maximum_retained_samples = observation.sample_bounds.maximum_retained_samples;
        Ok(Self {
            plan_identity: plan.identity,
            topology_digest: plan.topology_digest,
            independent_domain: plan.independent_domain.clone(),
            observation,
            maximum_produced_samples,
            maximum_retained_samples,
            observed_sample_count: 0,
            samples: VecDeque::with_capacity(maximum_retained_samples),
        })
    }

    pub fn push(
        &mut self,
        coordinate: RecoveredFieldSample,
        result: ObservationResult,
    ) -> Result<(), ObservationError> {
        if self.observed_sample_count
            >= u64::try_from(self.maximum_produced_samples).unwrap_or(u64::MAX)
        {
            return Err(ObservationError::Limit("history produced sample count"));
        }
        if !sample_matches_policy(&self.observation.spec.sampling, coordinate)? {
            return Err(ObservationError::Invalid(
                "history sample coordinate differs from its sampling policy".into(),
            ));
        }
        if !sample_matches_domain(&self.independent_domain, coordinate) {
            return Err(ObservationError::Invalid(
                "history sample coordinate differs from its independent domain".into(),
            ));
        }
        if let Some(previous) = self.samples.back() {
            if !coordinate_follows(previous.coordinate, coordinate) {
                return Err(ObservationError::Invalid(
                    "history sample coordinates must be strictly increasing in one domain".into(),
                ));
            }
        }
        validate_result_against_observation(
            &result,
            self.plan_identity,
            self.topology_digest,
            &self.observation,
        )?;
        let sample = ObservationHistorySample::new(coordinate, result)?;
        self.observed_sample_count = self
            .observed_sample_count
            .checked_add(1)
            .ok_or(ObservationError::Limit("history observed sample count"))?;
        if self.samples.len() == self.maximum_retained_samples {
            if matches!(
                self.observation.spec.storage.retention,
                RetentionPolicy::LatestSamples { .. }
            ) {
                self.samples.pop_front();
            } else {
                return Err(ObservationError::Limit("history retained sample count"));
            }
        }
        self.samples.push_back(sample);
        Ok(())
    }

    pub fn finish(self) -> Result<ObservationHistory, ObservationError> {
        if self.samples.is_empty() {
            return Err(ObservationError::Invalid(
                "history publication requires at least one retained sample".into(),
            ));
        }
        if let Some(expected) = exact_scheduled_sample_count(&self.observation.spec.sampling)? {
            if self.observed_sample_count
                != u64::try_from(expected)
                    .map_err(|_| ObservationError::Limit("history produced sample count"))?
            {
                return Err(ObservationError::Invalid(
                    "completed history does not contain its exact sampling schedule".into(),
                ));
            }
        }
        let samples = self.samples.into_iter().collect::<Vec<_>>();
        let mut manifest = ObservationHistoryManifest {
            schema_version: 1,
            identity: ObservationHistoryIdentity(Digest::sha256([])),
            plan_identity: self.plan_identity,
            topology_digest: self.topology_digest,
            independent_domain: self.independent_domain,
            observation_id: self.observation.spec.observation_id,
            quantity: self.observation.spec.quantity,
            location: self.observation.spec.output_location,
            value_shape: samples[0].result.value_shape.clone(),
            complex_representation: self.observation.spec.complex_representation,
            sampling: self.observation.spec.sampling,
            retention: self.observation.spec.storage.retention,
            partial_access: self.observation.spec.storage.partial_access,
            observed_sample_count: self.observed_sample_count,
            maximum_produced_samples: self.maximum_produced_samples,
            maximum_retained_samples: self.maximum_retained_samples,
            samples: samples
                .iter()
                .map(|sample| ObservationHistoryEntry {
                    sample: sample.identity,
                    coordinate: sample.coordinate,
                    result: sample.result.logical_identity,
                })
                .collect(),
        };
        manifest.identity = derive_history_identity(&manifest)?;
        ObservationHistory::from_manifest_and_samples(manifest, samples)
    }
}

fn validate_result_against_observation(
    result: &ObservationResult,
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    result.validate()?;
    let expected_shape = if observation.result_component_count == 1 {
        ObservationValueShape::Scalar
    } else {
        observation.spec.value_shape.clone()
    };
    if result.plan_identity != plan_identity
        || result.topology_digest != topology_digest
        || result.observation_id != observation.spec.observation_id
        || result.quantity != observation.spec.quantity
        || result.location != observation.spec.output_location
        || result.value_shape != expected_shape
        || result.complex_representation != observation.spec.complex_representation
        || result.partial_access != observation.spec.storage.partial_access
    {
        return Err(ObservationError::Identity(
            "history result differs from its resolved observation",
        ));
    }
    Ok(())
}
