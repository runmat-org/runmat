use std::collections::VecDeque;

use runmat_analysis_core::{IndependentDomain, ObservationId, RetentionPolicy, SamplingPolicy};
use runmat_execution::Digest;

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{
    ObservationError, ObservationResult, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity,
};
use super::checkpoint::{
    extend_execution_prefix_digest, extend_observed_prefix_digest, initial_execution_prefix_digest,
    initial_observed_prefix_digest, ObservationHistoryCheckpoint,
    ObservationHistoryCheckpointIdentity, ObservationHistoryStream,
};
use super::contract::{
    derive_history_identity, history_position_follows, ObservationHistory, ObservationHistoryEntry,
    ObservationHistoryIdentity, ObservationHistoryManifest, ObservationHistorySample,
};
use super::schedule::{
    exact_scheduled_coordinate_at, exact_scheduled_sample_count, sample_matches_domain,
    sample_matches_history_policy, validate_history_policy,
};

pub struct BoundedObservationHistory {
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    independent_domain: IndependentDomain,
    observation: ResolvedObservation,
    maximum_produced_samples: usize,
    maximum_retained_samples: usize,
    observed_sample_count: u64,
    stream: ObservationHistoryStream,
    retained_prefix_digest: Digest,
    observed_prefix_digest: Digest,
    retained_execution_prefix_digest: Digest,
    observed_execution_prefix_digest: Digest,
    samples: VecDeque<ObservationHistorySample>,
}

impl BoundedObservationHistory {
    pub fn new(
        plan: &ResolvedObservationPlan,
        observation_id: &ObservationId,
    ) -> Result<Self, ObservationError> {
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
        let stream = ObservationHistoryStream::new(
            plan.identity,
            plan.topology_digest,
            plan.independent_domain.clone(),
            &observation,
            maximum_produced_samples,
            maximum_retained_samples,
        )?;
        let initial_prefix = initial_observed_prefix_digest(stream.identity)?;
        let initial_execution_prefix = initial_execution_prefix_digest(stream.identity)?;
        Ok(Self {
            plan_identity: plan.identity,
            topology_digest: plan.topology_digest,
            independent_domain: plan.independent_domain.clone(),
            observation,
            maximum_produced_samples,
            maximum_retained_samples,
            observed_sample_count: 0,
            stream,
            retained_prefix_digest: initial_prefix,
            observed_prefix_digest: initial_prefix,
            retained_execution_prefix_digest: initial_execution_prefix,
            observed_execution_prefix_digest: initial_execution_prefix,
            // This is an upper bound, not an allocation forecast. Dynamic sampling can permit
            // many more samples than a completed solve actually contains; reserving the bound
            // eagerly can exhaust memory before the first observed sample is recovered.
            samples: VecDeque::new(),
        })
    }

    pub fn resume(
        plan: &ResolvedObservationPlan,
        observation_id: &ObservationId,
        checkpoint: ObservationHistoryCheckpoint,
    ) -> Result<Self, ObservationError> {
        checkpoint.validate()?;
        let mut history = Self::new(plan, observation_id)?;
        if checkpoint.manifest().stream != history.stream {
            return Err(ObservationError::Identity(
                "history checkpoint differs from the requested resolved observation",
            ));
        }
        history.observed_sample_count = checkpoint.manifest().observed_sample_count;
        history.retained_prefix_digest = checkpoint.manifest().retained_prefix_digest;
        history.observed_prefix_digest = checkpoint.manifest().observed_prefix_digest;
        history.retained_execution_prefix_digest =
            checkpoint.manifest().retained_execution_prefix_digest;
        history.observed_execution_prefix_digest =
            checkpoint.manifest().observed_execution_prefix_digest;
        history.samples = checkpoint.samples().iter().cloned().collect();
        Ok(history)
    }

    pub fn checkpoint(
        &self,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
    ) -> Result<ObservationHistoryCheckpoint, ObservationError> {
        if self.observed_sample_count == 0 {
            return Err(ObservationError::Invalid(
                "history checkpoint requires at least one observed sample".into(),
            ));
        }
        ObservationHistoryCheckpoint::new(
            self.stream.clone(),
            predecessor,
            self.observed_sample_count,
            self.retained_prefix_digest,
            self.observed_prefix_digest,
            (
                self.retained_execution_prefix_digest,
                self.observed_execution_prefix_digest,
            ),
            self.samples.iter().cloned().collect(),
        )
    }

    pub fn push(
        &mut self,
        coordinate: RecoveredFieldSample,
        result: ObservationResult,
    ) -> Result<(), ObservationError> {
        self.push_sample(ObservationHistorySample::new(coordinate, result)?)
    }

    pub fn push_event(&mut self, sample: ObservationHistorySample) -> Result<(), ObservationError> {
        if sample.event_attribution.is_none() {
            return Err(ObservationError::Invalid(
                "event history requires source attribution".into(),
            ));
        }
        self.push_sample(sample)
    }

    fn push_sample(&mut self, sample: ObservationHistorySample) -> Result<(), ObservationError> {
        sample.validate()?;
        let coordinate = sample.coordinate;
        let event_attribution = sample.event_attribution;
        if self.observed_sample_count
            >= u64::try_from(self.maximum_produced_samples).unwrap_or(u64::MAX)
        {
            return Err(ObservationError::Limit("history produced sample count"));
        }
        let ordinal = usize::try_from(self.observed_sample_count)
            .map_err(|_| ObservationError::Limit("history observed sample count"))?;
        if exact_scheduled_coordinate_at(&self.observation.spec.sampling, ordinal)?
            .is_some_and(|expected| expected != coordinate)
        {
            return Err(ObservationError::Invalid(
                "history sample is not the next exact scheduled coordinate".into(),
            ));
        }
        let policy_matches = sample_matches_history_policy(
            &self.observation.spec.sampling,
            coordinate,
            event_attribution,
        )?;
        if !policy_matches {
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
            if !history_position_follows(
                previous.coordinate,
                previous.event_attribution,
                coordinate,
                event_attribution,
            ) {
                return Err(ObservationError::Invalid(
                    "history sample coordinates must be strictly increasing in one domain".into(),
                ));
            }
        }
        validate_result_against_observation(
            &sample.result,
            self.plan_identity,
            self.topology_digest,
            &self.observation,
        )?;
        if self.samples.len() == self.maximum_retained_samples {
            if matches!(
                self.observation.spec.storage.retention,
                RetentionPolicy::LatestSamples { .. }
            ) {
                let evicted = self.samples.pop_front().ok_or_else(|| {
                    ObservationError::Invalid(
                        "history retention accounting lost its oldest sample".into(),
                    )
                })?;
                self.retained_prefix_digest =
                    extend_observed_prefix_digest(self.retained_prefix_digest, evicted.identity)?;
                self.retained_execution_prefix_digest = extend_execution_prefix_digest(
                    self.retained_execution_prefix_digest,
                    evicted.identity,
                    evicted.result.partition_evidence.identity,
                )?;
            } else {
                return Err(ObservationError::Limit("history retained sample count"));
            }
        }
        self.observed_prefix_digest =
            extend_observed_prefix_digest(self.observed_prefix_digest, sample.identity)?;
        self.observed_execution_prefix_digest = extend_execution_prefix_digest(
            self.observed_execution_prefix_digest,
            sample.identity,
            sample.result.partition_evidence.identity,
        )?;
        self.observed_sample_count = self
            .observed_sample_count
            .checked_add(1)
            .ok_or(ObservationError::Limit("history observed sample count"))?;
        self.samples.push_back(sample);
        Ok(())
    }

    pub fn finish(self) -> Result<ObservationHistory, ObservationError> {
        if self.samples.is_empty()
            && !matches!(self.observation.spec.sampling, SamplingPolicy::Event(_))
        {
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
        let value_shape = samples.first().map_or_else(
            || self.observation.result_value_shape.clone(),
            |sample| sample.result.value_shape.clone(),
        );
        let mut manifest = ObservationHistoryManifest {
            schema_version: 1,
            identity: ObservationHistoryIdentity(Digest::sha256([])),
            stream_identity: self.stream.identity,
            plan_identity: self.plan_identity,
            topology_digest: self.topology_digest,
            independent_domain: self.independent_domain,
            observation_id: self.observation.spec.observation_id,
            quantity: self.observation.spec.quantity,
            location: self.observation.spec.output_location,
            value_shape,
            complex_representation: self.observation.spec.complex_representation,
            sampling: self.observation.spec.sampling,
            retention: self.observation.spec.storage.retention,
            partial_access: self.observation.spec.storage.partial_access,
            observed_sample_count: self.observed_sample_count,
            retained_prefix_digest: self.retained_prefix_digest,
            observed_prefix_digest: self.observed_prefix_digest,
            maximum_produced_samples: self.maximum_produced_samples,
            maximum_retained_samples: self.maximum_retained_samples,
            samples: samples
                .iter()
                .map(|sample| ObservationHistoryEntry {
                    sample: sample.identity,
                    coordinate: sample.coordinate,
                    event_attribution: sample.event_attribution,
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
    let expected_shape = &observation.result_value_shape;
    if result.plan_identity != plan_identity
        || result.topology_digest != topology_digest
        || result.observation_id != observation.spec.observation_id
        || result.quantity != observation.spec.quantity
        || result.location != observation.spec.output_location
        || &result.value_shape != expected_shape
        || result.complex_representation != observation.spec.complex_representation
        || result.partial_access != observation.spec.storage.partial_access
    {
        return Err(ObservationError::Identity(
            "history result differs from its resolved observation",
        ));
    }
    Ok(())
}
