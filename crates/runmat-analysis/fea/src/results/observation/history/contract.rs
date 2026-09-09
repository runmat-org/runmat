use runmat_analysis_core::{
    FieldLocation, IndependentDomain, ObservationId, ObservationQuantity, ObservationValueShape,
    PartialAccessPolicy, RetentionPolicy, SamplingPolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::results::field::RecoveredFieldSample;

use super::super::contract::{
    ObservationError, ObservationResult, ObservationResultIdentity,
    ResolvedObservationPlanIdentity, OBSERVATION_LIMITS, OBSERVATION_PREFIX,
};
use super::schedule::{
    coordinate_follows, exact_scheduled_sample_count, resolve_history_sample_bounds,
    sample_matches_domain, sample_matches_policy, validate_coordinate, validate_history_policy,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationHistorySampleIdentity(Digest);

impl ObservationHistorySampleIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationHistoryIdentity(pub(super) Digest);

impl ObservationHistoryIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationHistorySampleWire")]
pub struct ObservationHistorySample {
    pub schema_version: u16,
    pub identity: ObservationHistorySampleIdentity,
    pub coordinate: RecoveredFieldSample,
    pub result: ObservationResult,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistorySampleWire {
    schema_version: u16,
    identity: ObservationHistorySampleIdentity,
    coordinate: RecoveredFieldSample,
    result: ObservationResult,
}

impl TryFrom<ObservationHistorySampleWire> for ObservationHistorySample {
    type Error = ObservationError;

    fn try_from(wire: ObservationHistorySampleWire) -> Result<Self, Self::Error> {
        let sample = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            coordinate: wire.coordinate,
            result: wire.result,
        };
        sample.validate()?;
        Ok(sample)
    }
}

impl ObservationHistorySample {
    pub fn new(
        coordinate: RecoveredFieldSample,
        result: ObservationResult,
    ) -> Result<Self, ObservationError> {
        validate_coordinate(coordinate)?;
        result.validate()?;
        let identity = derive_sample_identity(coordinate, result.logical_identity)?;
        Ok(Self {
            schema_version: 1,
            identity,
            coordinate,
            result,
        })
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        validate_coordinate(self.coordinate)?;
        self.result.validate()?;
        if self.schema_version != 1
            || self.identity
                != derive_sample_identity(self.coordinate, self.result.logical_identity)?
        {
            return Err(ObservationError::Identity(
                "history sample identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-history-sample",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        Ok(decode_contract(
            OBSERVATION_PREFIX,
            "observation-history-sample",
            bytes,
            OBSERVATION_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationHistoryEntry {
    pub sample: ObservationHistorySampleIdentity,
    pub coordinate: RecoveredFieldSample,
    pub result: ObservationResultIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationHistoryManifestWire")]
pub struct ObservationHistoryManifest {
    pub schema_version: u16,
    pub identity: ObservationHistoryIdentity,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub independent_domain: IndependentDomain,
    pub observation_id: ObservationId,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub sampling: SamplingPolicy,
    pub retention: RetentionPolicy,
    pub partial_access: PartialAccessPolicy,
    pub observed_sample_count: u64,
    pub maximum_produced_samples: usize,
    pub maximum_retained_samples: usize,
    pub samples: Vec<ObservationHistoryEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryManifestWire {
    schema_version: u16,
    identity: ObservationHistoryIdentity,
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    independent_domain: IndependentDomain,
    observation_id: ObservationId,
    quantity: ObservationQuantity,
    location: FieldLocation,
    value_shape: ObservationValueShape,
    sampling: SamplingPolicy,
    retention: RetentionPolicy,
    partial_access: PartialAccessPolicy,
    observed_sample_count: u64,
    maximum_produced_samples: usize,
    maximum_retained_samples: usize,
    samples: Vec<ObservationHistoryEntry>,
}

impl TryFrom<ObservationHistoryManifestWire> for ObservationHistoryManifest {
    type Error = ObservationError;

    fn try_from(wire: ObservationHistoryManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            plan_identity: wire.plan_identity,
            topology_digest: wire.topology_digest,
            independent_domain: wire.independent_domain,
            observation_id: wire.observation_id,
            quantity: wire.quantity,
            location: wire.location,
            value_shape: wire.value_shape,
            sampling: wire.sampling,
            retention: wire.retention,
            partial_access: wire.partial_access,
            observed_sample_count: wire.observed_sample_count,
            maximum_produced_samples: wire.maximum_produced_samples,
            maximum_retained_samples: wire.maximum_retained_samples,
            samples: wire.samples,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl ObservationHistoryManifest {
    pub fn validate(&self) -> Result<(), ObservationError> {
        let expected_bounds = resolve_history_sample_bounds(
            &self.sampling,
            &self.retention,
            self.maximum_produced_samples,
        )?;
        let retained_count = u64::try_from(self.samples.len())
            .map_err(|_| ObservationError::Limit("history retained sample count"))?;
        let maximum_produced = u64::try_from(self.maximum_produced_samples)
            .map_err(|_| ObservationError::Limit("history produced sample count"))?;
        if self.schema_version != 1
            || self.maximum_produced_samples == 0
            || self.maximum_retained_samples == 0
            || self.maximum_retained_samples > self.maximum_produced_samples
            || self.maximum_produced_samples != expected_bounds.maximum_produced_samples
            || self.maximum_retained_samples != expected_bounds.maximum_retained_samples
            || self.samples.is_empty()
            || self.samples.len() > self.maximum_retained_samples
            || self.observed_sample_count < retained_count
        {
            return Err(ObservationError::Invalid(
                "history manifest has invalid schema, bounds, or sample counts".into(),
            ));
        }
        validate_history_policy(&self.sampling, &self.retention)?;
        if !runmat_analysis_core::observation_sampling_matches_domain(
            &self.independent_domain,
            &self.sampling,
        ) {
            return Err(ObservationError::Invalid(
                "history sampling policy differs from its independent domain".into(),
            ));
        }
        if self.observed_sample_count > maximum_produced {
            return Err(ObservationError::Limit("history produced sample count"));
        }
        if exact_scheduled_sample_count(&self.sampling)?
            .is_some_and(|expected| u64::try_from(expected) != Ok(self.observed_sample_count))
        {
            return Err(ObservationError::Invalid(
                "completed history does not contain its exact sampling schedule".into(),
            ));
        }
        let expected_present = match self.retention {
            RetentionPolicy::LatestSamples { .. } => usize::try_from(self.observed_sample_count)
                .unwrap_or(usize::MAX)
                .min(self.maximum_retained_samples),
            _ => usize::try_from(self.observed_sample_count).unwrap_or(usize::MAX),
        };
        if self.samples.len() != expected_present {
            return Err(ObservationError::Invalid(
                "history manifest sample inventory differs from its retention policy".into(),
            ));
        }
        validate_entry_order(&self.samples)?;
        if self.identity != derive_history_identity(self)? {
            return Err(ObservationError::Identity(
                "history manifest identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-history-manifest",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        Ok(decode_contract(
            OBSERVATION_PREFIX,
            "observation-history-manifest",
            bytes,
            OBSERVATION_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObservationHistory {
    manifest: ObservationHistoryManifest,
    samples: Vec<ObservationHistorySample>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationHistoryWire {
    manifest: ObservationHistoryManifest,
    samples: Vec<ObservationHistorySample>,
}

impl ObservationHistory {
    pub fn manifest(&self) -> &ObservationHistoryManifest {
        &self.manifest
    }

    pub fn samples(&self) -> &[ObservationHistorySample] {
        &self.samples
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-history",
            &ObservationHistoryWire {
                manifest: self.manifest.clone(),
                samples: self.samples.clone(),
            },
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let wire: ObservationHistoryWire = decode_contract(
            OBSERVATION_PREFIX,
            "observation-history",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Self::from_manifest_and_samples(wire.manifest, wire.samples)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        validate_manifest_samples(&self.manifest, &self.samples)
    }

    pub fn from_manifest_and_samples(
        manifest: ObservationHistoryManifest,
        samples: Vec<ObservationHistorySample>,
    ) -> Result<Self, ObservationError> {
        validate_manifest_samples(&manifest, &samples)?;
        Ok(Self { manifest, samples })
    }
}

fn validate_manifest_samples(
    manifest: &ObservationHistoryManifest,
    samples: &[ObservationHistorySample],
) -> Result<(), ObservationError> {
    manifest.validate()?;
    if manifest.samples.len() != samples.len()
        || manifest.samples.iter().zip(samples).any(|(entry, sample)| {
            entry.sample != sample.identity
                || entry.coordinate != sample.coordinate
                || entry.result != sample.result.logical_identity
        })
    {
        return Err(ObservationError::Identity(
            "history samples differ from their manifest",
        ));
    }
    for sample in samples {
        validate_sample_against_manifest(sample, manifest)?;
    }
    Ok(())
}

fn validate_entry_order(entries: &[ObservationHistoryEntry]) -> Result<(), ObservationError> {
    for entry in entries {
        validate_coordinate(entry.coordinate)?;
        if entry.sample != derive_sample_identity(entry.coordinate, entry.result)? {
            return Err(ObservationError::Identity(
                "history manifest entry identity does not converge",
            ));
        }
    }
    if entries
        .windows(2)
        .any(|pair| !coordinate_follows(pair[0].coordinate, pair[1].coordinate))
    {
        return Err(ObservationError::Invalid(
            "history manifest coordinates must be strictly increasing in one domain".into(),
        ));
    }
    Ok(())
}

fn validate_sample_against_manifest(
    sample: &ObservationHistorySample,
    manifest: &ObservationHistoryManifest,
) -> Result<(), ObservationError> {
    sample.validate()?;
    if sample.result.plan_identity != manifest.plan_identity
        || sample.result.topology_digest != manifest.topology_digest
        || sample.result.observation_id != manifest.observation_id
        || sample.result.quantity != manifest.quantity
        || sample.result.location != manifest.location
        || sample.result.value_shape != manifest.value_shape
        || sample.result.partial_access != manifest.partial_access
        || !sample_matches_policy(&manifest.sampling, sample.coordinate)?
        || !sample_matches_domain(&manifest.independent_domain, sample.coordinate)
    {
        return Err(ObservationError::Identity(
            "history sample differs from its manifest",
        ));
    }
    Ok(())
}

fn derive_sample_identity(
    coordinate: RecoveredFieldSample,
    result: ObservationResultIdentity,
) -> Result<ObservationHistorySampleIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "observation-history-sample-identity",
        &(coordinate, result),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationHistorySampleIdentity(Digest::sha256(bytes)))
}

pub(super) fn derive_history_identity(
    manifest: &ObservationHistoryManifest,
) -> Result<ObservationHistoryIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "observation-history-identity",
        &(
            manifest.schema_version,
            manifest.plan_identity,
            manifest.topology_digest,
            &manifest.independent_domain,
            &manifest.observation_id,
            manifest.quantity,
            manifest.location,
            &manifest.value_shape,
            &manifest.sampling,
            &manifest.retention,
            manifest.partial_access,
            manifest.observed_sample_count,
            manifest.maximum_produced_samples,
            manifest.maximum_retained_samples,
            &manifest.samples,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationHistoryIdentity(Digest::sha256(bytes)))
}
