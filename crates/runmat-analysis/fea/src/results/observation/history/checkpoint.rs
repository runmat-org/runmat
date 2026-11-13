use super::super::{ObservationPartitionEvidence, ObservationPartitionEvidenceIdentity};
use runmat_analysis_core::{
    ComplexRepresentation, FieldLocation, IndependentDomain, ObservationId, ObservationQuantity,
    ObservationValueShape, PartialAccessPolicy, RetentionPolicy, SamplingPolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::super::contract::{
    ObservationError, ResolvedObservation, ResolvedObservationPlanIdentity, OBSERVATION_LIMITS,
    OBSERVATION_PREFIX,
};
use super::contract::{
    validate_entry_order, ObservationHistoryEntry, ObservationHistoryManifest,
    ObservationHistorySample, ObservationHistorySampleIdentity,
};
use super::schedule::{
    exact_scheduled_coordinate_at, resolve_history_sample_bounds, sample_matches_domain,
    sample_matches_history_policy, validate_history_policy,
};

const CHECKPOINT_PREFIX: &[u8] = b"runmat-observation-history-checkpoint/v3\0";
const CHECKPOINT_SCHEMA_VERSION: u16 = 3;

mod identity;
mod lineage;
use identity::derive_checkpoint_identity;
pub(super) use identity::{extend_execution_prefix_digest, initial_execution_prefix_digest};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationHistoryStreamIdentity(Digest);

impl ObservationHistoryStreamIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationHistoryCheckpointIdentity(Digest);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationHistoryStreamWire")]
pub struct ObservationHistoryStream {
    pub schema_version: u16,
    pub identity: ObservationHistoryStreamIdentity,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub independent_domain: IndependentDomain,
    pub observation_id: ObservationId,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub complex_representation: ComplexRepresentation,
    pub sampling: SamplingPolicy,
    pub retention: RetentionPolicy,
    pub partial_access: PartialAccessPolicy,
    pub maximum_produced_samples: usize,
    pub maximum_retained_samples: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryStreamWire {
    schema_version: u16,
    identity: ObservationHistoryStreamIdentity,
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    independent_domain: IndependentDomain,
    observation_id: ObservationId,
    quantity: ObservationQuantity,
    location: FieldLocation,
    value_shape: ObservationValueShape,
    complex_representation: ComplexRepresentation,
    sampling: SamplingPolicy,
    retention: RetentionPolicy,
    partial_access: PartialAccessPolicy,
    maximum_produced_samples: usize,
    maximum_retained_samples: usize,
}

impl TryFrom<ObservationHistoryStreamWire> for ObservationHistoryStream {
    type Error = ObservationError;

    fn try_from(wire: ObservationHistoryStreamWire) -> Result<Self, Self::Error> {
        let stream = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            plan_identity: wire.plan_identity,
            topology_digest: wire.topology_digest,
            independent_domain: wire.independent_domain,
            observation_id: wire.observation_id,
            quantity: wire.quantity,
            location: wire.location,
            value_shape: wire.value_shape,
            complex_representation: wire.complex_representation,
            sampling: wire.sampling,
            retention: wire.retention,
            partial_access: wire.partial_access,
            maximum_produced_samples: wire.maximum_produced_samples,
            maximum_retained_samples: wire.maximum_retained_samples,
        };
        stream.validate()?;
        Ok(stream)
    }
}

impl ObservationHistoryStream {
    pub(crate) fn new(
        plan_identity: ResolvedObservationPlanIdentity,
        topology_digest: Digest,
        independent_domain: IndependentDomain,
        observation: &ResolvedObservation,
        maximum_produced_samples: usize,
        maximum_retained_samples: usize,
    ) -> Result<Self, ObservationError> {
        let mut stream = Self {
            schema_version: 1,
            identity: ObservationHistoryStreamIdentity(Digest::sha256([])),
            plan_identity,
            topology_digest,
            independent_domain,
            observation_id: observation.spec.observation_id.clone(),
            quantity: observation.spec.quantity,
            location: observation.spec.output_location,
            value_shape: observation.result_value_shape.clone(),
            complex_representation: observation.spec.complex_representation,
            sampling: observation.spec.sampling.clone(),
            retention: observation.spec.storage.retention.clone(),
            partial_access: observation.spec.storage.partial_access,
            maximum_produced_samples,
            maximum_retained_samples,
        };
        stream.identity = derive_stream_identity(&stream)?;
        stream.validate()?;
        Ok(stream)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        validate_history_policy(&self.sampling, &self.retention)?;
        let bounds = resolve_history_sample_bounds(
            &self.sampling,
            &self.retention,
            self.maximum_produced_samples,
        )?;
        if self.schema_version != 1
            || self.maximum_produced_samples == 0
            || self.maximum_retained_samples == 0
            || bounds.maximum_produced_samples != self.maximum_produced_samples
            || bounds.maximum_retained_samples != self.maximum_retained_samples
            || !runmat_analysis_core::observation_sampling_matches_domain(
                &self.independent_domain,
                &self.sampling,
            )
        {
            return Err(ObservationError::Invalid(
                "history stream has invalid schema, domain, or sample bounds".into(),
            ));
        }
        if self.identity != derive_stream_identity(self)? {
            return Err(ObservationError::Identity(
                "history stream identity does not converge",
            ));
        }
        Ok(())
    }

    pub(super) fn from_history_manifest(
        manifest: &ObservationHistoryManifest,
    ) -> Result<Self, ObservationError> {
        let stream = Self {
            schema_version: 1,
            identity: manifest.stream_identity,
            plan_identity: manifest.plan_identity,
            topology_digest: manifest.topology_digest,
            independent_domain: manifest.independent_domain.clone(),
            observation_id: manifest.observation_id.clone(),
            quantity: manifest.quantity,
            location: manifest.location,
            value_shape: manifest.value_shape.clone(),
            complex_representation: manifest.complex_representation,
            sampling: manifest.sampling.clone(),
            retention: manifest.retention.clone(),
            partial_access: manifest.partial_access,
            maximum_produced_samples: manifest.maximum_produced_samples,
            maximum_retained_samples: manifest.maximum_retained_samples,
        };
        stream.validate()?;
        Ok(stream)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "ObservationHistoryCheckpointManifestWire"
)]
pub struct ObservationHistoryCheckpointManifest {
    pub schema_version: u16,
    pub identity: ObservationHistoryCheckpointIdentity,
    pub stream: ObservationHistoryStream,
    pub predecessor: Option<ObservationHistoryCheckpointIdentity>,
    pub observed_sample_count: u64,
    pub retained_prefix_digest: Digest,
    pub observed_prefix_digest: Digest,
    pub samples: Vec<ObservationHistoryEntry>,
    pub retained_execution_prefix_digest: Digest,
    pub observed_execution_prefix_digest: Digest,
    pub partition_evidence: Vec<ObservationPartitionEvidence>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryCheckpointManifestWire {
    schema_version: u16,
    identity: ObservationHistoryCheckpointIdentity,
    stream: ObservationHistoryStream,
    predecessor: Option<ObservationHistoryCheckpointIdentity>,
    observed_sample_count: u64,
    retained_prefix_digest: Digest,
    observed_prefix_digest: Digest,
    samples: Vec<ObservationHistoryEntry>,
    retained_execution_prefix_digest: Digest,
    observed_execution_prefix_digest: Digest,
    partition_evidence: Vec<ObservationPartitionEvidence>,
}

impl TryFrom<ObservationHistoryCheckpointManifestWire> for ObservationHistoryCheckpointManifest {
    type Error = ObservationError;

    fn try_from(wire: ObservationHistoryCheckpointManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            stream: wire.stream,
            predecessor: wire.predecessor,
            observed_sample_count: wire.observed_sample_count,
            retained_prefix_digest: wire.retained_prefix_digest,
            observed_prefix_digest: wire.observed_prefix_digest,
            samples: wire.samples,
            retained_execution_prefix_digest: wire.retained_execution_prefix_digest,
            observed_execution_prefix_digest: wire.observed_execution_prefix_digest,
            partition_evidence: wire.partition_evidence,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl ObservationHistoryCheckpointManifest {
    pub fn validate(&self) -> Result<(), ObservationError> {
        validate_checkpoint_manifest(self)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            CHECKPOINT_PREFIX,
            "observation-history-checkpoint-manifest",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let manifest: Self = decode_contract(
            CHECKPOINT_PREFIX,
            "observation-history-checkpoint-manifest",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationHistoryCheckpointWire")]
pub struct ObservationHistoryCheckpoint {
    manifest: ObservationHistoryCheckpointManifest,
    samples: Vec<ObservationHistorySample>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryCheckpointWire {
    manifest: ObservationHistoryCheckpointManifest,
    samples: Vec<ObservationHistorySample>,
}

impl TryFrom<ObservationHistoryCheckpointWire> for ObservationHistoryCheckpoint {
    type Error = ObservationError;

    fn try_from(wire: ObservationHistoryCheckpointWire) -> Result<Self, Self::Error> {
        Self::from_manifest_and_samples(wire.manifest, wire.samples)
    }
}

impl ObservationHistoryCheckpoint {
    pub(super) fn new(
        stream: ObservationHistoryStream,
        predecessor: Option<ObservationHistoryCheckpointIdentity>,
        observed_sample_count: u64,
        retained_prefix_digest: Digest,
        observed_prefix_digest: Digest,
        execution_prefixes: (Digest, Digest),
        samples: Vec<ObservationHistorySample>,
    ) -> Result<Self, ObservationError> {
        let entries = samples.iter().map(history_entry).collect();
        let mut manifest = ObservationHistoryCheckpointManifest {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            identity: ObservationHistoryCheckpointIdentity(Digest::sha256([])),
            stream,
            predecessor,
            observed_sample_count,
            retained_prefix_digest,
            observed_prefix_digest,
            samples: entries,
            retained_execution_prefix_digest: execution_prefixes.0,
            observed_execution_prefix_digest: execution_prefixes.1,
            partition_evidence: samples
                .iter()
                .map(|sample| sample.result.partition_evidence.clone())
                .collect(),
        };
        manifest.identity = derive_checkpoint_identity(&manifest)?;
        Self::from_manifest_and_samples(manifest, samples)
    }

    pub fn manifest(&self) -> &ObservationHistoryCheckpointManifest {
        &self.manifest
    }

    pub fn samples(&self) -> &[ObservationHistorySample] {
        &self.samples
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        validate_checkpoint(&self.manifest, &self.samples)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            CHECKPOINT_PREFIX,
            "observation-history-checkpoint",
            &ObservationHistoryCheckpointWire {
                manifest: self.manifest.clone(),
                samples: self.samples.clone(),
            },
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let wire: ObservationHistoryCheckpointWire = decode_contract(
            CHECKPOINT_PREFIX,
            "observation-history-checkpoint",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Self::try_from(wire)
    }

    pub fn from_manifest_and_samples(
        manifest: ObservationHistoryCheckpointManifest,
        samples: Vec<ObservationHistorySample>,
    ) -> Result<Self, ObservationError> {
        validate_checkpoint(&manifest, &samples)?;
        Ok(Self { manifest, samples })
    }
}

pub(super) fn initial_observed_prefix_digest(
    stream: ObservationHistoryStreamIdentity,
) -> Result<Digest, ObservationError> {
    Ok(Digest::sha256(encode_contract(
        OBSERVATION_PREFIX,
        "observation-history-prefix-initial",
        &stream,
        OBSERVATION_LIMITS,
    )?))
}

pub(super) fn extend_observed_prefix_digest(
    prefix: Digest,
    sample: ObservationHistorySampleIdentity,
) -> Result<Digest, ObservationError> {
    Ok(Digest::sha256(encode_contract(
        OBSERVATION_PREFIX,
        "observation-history-prefix-next",
        &(prefix, sample),
        OBSERVATION_LIMITS,
    )?))
}

fn validate_checkpoint(
    manifest: &ObservationHistoryCheckpointManifest,
    samples: &[ObservationHistorySample],
) -> Result<(), ObservationError> {
    manifest.validate()?;
    let expected_retained = match manifest.stream.retention {
        RetentionPolicy::LatestSamples { .. } => manifest.observed_sample_count.min(
            u64::try_from(manifest.stream.maximum_retained_samples)
                .map_err(|_| ObservationError::Limit("history retained sample count"))?,
        ),
        _ => manifest.observed_sample_count,
    };
    if u64::try_from(samples.len()) != Ok(expected_retained)
        || manifest.samples.len() != samples.len()
    {
        return Err(ObservationError::Invalid(
            "history checkpoint payload differs from its retained sample count".into(),
        ));
    }
    for ((entry, evidence), sample) in manifest
        .samples
        .iter()
        .zip(&manifest.partition_evidence)
        .zip(samples)
    {
        sample.validate()?;
        if entry != &history_entry(sample) || evidence != &sample.result.partition_evidence {
            return Err(ObservationError::Identity(
                "history checkpoint samples differ from their manifest",
            ));
        }
        validate_sample_against_stream(sample, &manifest.stream)?;
    }
    Ok(())
}

fn validate_checkpoint_manifest(
    manifest: &ObservationHistoryCheckpointManifest,
) -> Result<(), ObservationError> {
    manifest.stream.validate()?;
    let retained_count = u64::try_from(manifest.samples.len())
        .map_err(|_| ObservationError::Limit("history checkpoint retained samples"))?;
    let maximum_produced = u64::try_from(manifest.stream.maximum_produced_samples)
        .map_err(|_| ObservationError::Limit("history produced sample count"))?;
    let maximum_retained = u64::try_from(manifest.stream.maximum_retained_samples)
        .map_err(|_| ObservationError::Limit("history retained sample count"))?;
    let expected_retained = match manifest.stream.retention {
        RetentionPolicy::LatestSamples { .. } => {
            manifest.observed_sample_count.min(maximum_retained)
        }
        _ => manifest.observed_sample_count,
    };
    if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION
        || manifest.observed_sample_count == 0
        || manifest.observed_sample_count > maximum_produced
        || retained_count != expected_retained
    {
        return Err(ObservationError::Invalid(
            "history checkpoint has invalid schema or sample counts".into(),
        ));
    }
    validate_entry_order(&manifest.samples)?;
    identity::validate_execution_prefix(manifest)?;
    validate_observed_prefix(
        &manifest.stream,
        manifest.observed_sample_count,
        manifest.retained_prefix_digest,
        manifest.observed_prefix_digest,
        &manifest.samples,
    )?;
    if manifest.identity != derive_checkpoint_identity(manifest)?
        || manifest.predecessor == Some(manifest.identity)
    {
        return Err(ObservationError::Identity(
            "history checkpoint identity or prefix does not converge",
        ));
    }
    Ok(())
}

pub(super) fn validate_observed_prefix(
    stream: &ObservationHistoryStream,
    observed_sample_count: u64,
    retained_prefix_digest: Digest,
    observed_prefix_digest: Digest,
    entries: &[ObservationHistoryEntry],
) -> Result<(), ObservationError> {
    let retained_count = u64::try_from(entries.len())
        .map_err(|_| ObservationError::Limit("history retained sample count"))?;
    let evicted_count = observed_sample_count
        .checked_sub(retained_count)
        .ok_or_else(|| {
            ObservationError::Invalid("history retained count exceeds observed".into())
        })?;
    if evicted_count == 0
        && retained_prefix_digest != initial_observed_prefix_digest(stream.identity)?
    {
        return Err(ObservationError::Identity(
            "history has an invalid initial observed prefix",
        ));
    }
    let base = usize::try_from(evicted_count)
        .map_err(|_| ObservationError::Limit("history sample ordinal"))?;
    let mut prefix = retained_prefix_digest;
    for (index, entry) in entries.iter().enumerate() {
        let ordinal = base
            .checked_add(index)
            .ok_or(ObservationError::Limit("history sample ordinal"))?;
        if exact_scheduled_coordinate_at(&stream.sampling, ordinal)?
            .is_some_and(|expected| expected != entry.coordinate)
        {
            return Err(ObservationError::Identity(
                "history omits or reorders an exact scheduled sample",
            ));
        }
        prefix = extend_observed_prefix_digest(prefix, entry.sample)?;
    }
    if prefix != observed_prefix_digest {
        return Err(ObservationError::Identity(
            "history observed prefix does not converge",
        ));
    }
    Ok(())
}

fn validate_sample_against_stream(
    sample: &ObservationHistorySample,
    stream: &ObservationHistoryStream,
) -> Result<(), ObservationError> {
    if sample.result.plan_identity != stream.plan_identity
        || sample.result.topology_digest != stream.topology_digest
        || sample.result.observation_id != stream.observation_id
        || sample.result.quantity != stream.quantity
        || sample.result.location != stream.location
        || sample.result.value_shape != stream.value_shape
        || sample.result.complex_representation != stream.complex_representation
        || sample.result.partial_access != stream.partial_access
        || !sample_matches_history_policy(
            &stream.sampling,
            sample.coordinate,
            sample.event_attribution,
        )?
        || !sample_matches_domain(&stream.independent_domain, sample.coordinate)
    {
        return Err(ObservationError::Identity(
            "history checkpoint sample differs from its stream",
        ));
    }
    Ok(())
}

fn history_entry(sample: &ObservationHistorySample) -> ObservationHistoryEntry {
    ObservationHistoryEntry {
        sample: sample.identity,
        coordinate: sample.coordinate,
        event_attribution: sample.event_attribution,
        result: sample.result.logical_identity,
    }
}

fn derive_stream_identity(
    stream: &ObservationHistoryStream,
) -> Result<ObservationHistoryStreamIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "observation-history-stream-identity",
        &(
            stream.schema_version,
            stream.plan_identity,
            stream.topology_digest,
            &stream.independent_domain,
            &stream.observation_id,
            stream.quantity,
            stream.location,
            &stream.value_shape,
            stream.complex_representation,
            &stream.sampling,
            &stream.retention,
            stream.partial_access,
            stream.maximum_produced_samples,
            stream.maximum_retained_samples,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationHistoryStreamIdentity(Digest::sha256(bytes)))
}
