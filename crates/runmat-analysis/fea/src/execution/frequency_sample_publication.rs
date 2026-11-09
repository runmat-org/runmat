//! Path-free, physics-neutral publication of a complete independent-frequency inventory.
//!
//! The caller supplies authenticated aggregate and per-entry numerical authority. This contract
//! binds that authority to recovered catalogs; it does not invent receipts, load numerical
//! arrays, execute a solver, or establish formulation qualification.
use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::frequency::{FrequencyResponsePlan, FrequencyResponsePlanEntry};
use crate::results::field::{RecoveredFieldArtifactManifest, RecoveredFieldSample};
use crate::results::observation::{ObservationArtifactManifest, ObservationFieldSource};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity, SolverResultIdentity,
    StepIdentity,
};

const PREFIX: &[u8] = b"runmat-fea-frequency-sample-publication/v1\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FrequencySamplePublicationIdentity(Digest);

impl FrequencySamplePublicationIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Numerical identity only: neither physical values nor transport locations are retained here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencySampleNumericalEntry {
    pub entry: FrequencyResponsePlanEntry,
    pub workload: AnalysisWorkloadIdentity,
    pub source: FieldRecoverySource,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "AuthorityWire")]
pub struct FrequencySamplePublicationAuthority {
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    producer: StepIdentity,
    topology: FieldTopologyIdentity,
    plan: FrequencyResponsePlan,
    entries: Vec<FrequencySampleNumericalEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorityWire {
    workload: AnalysisWorkloadIdentity,
    result: SolverResultIdentity,
    producer: StepIdentity,
    topology: FieldTopologyIdentity,
    plan: FrequencyResponsePlan,
    entries: Vec<FrequencySampleNumericalEntry>,
}

impl TryFrom<AuthorityWire> for FrequencySamplePublicationAuthority {
    type Error = AnalysisExecutionError;

    fn try_from(wire: AuthorityWire) -> Result<Self, Self::Error> {
        Self::new(
            wire.workload,
            wire.result,
            wire.producer,
            wire.topology,
            wire.plan,
            wire.entries,
        )
    }
}

impl FrequencySamplePublicationAuthority {
    pub fn new(
        workload: AnalysisWorkloadIdentity,
        result: SolverResultIdentity,
        producer: StepIdentity,
        topology: FieldTopologyIdentity,
        plan: FrequencyResponsePlan,
        entries: Vec<FrequencySampleNumericalEntry>,
    ) -> AnalysisExecutionResult<Self> {
        let authority = Self {
            workload,
            result,
            producer,
            topology,
            plan,
            entries,
        };
        authority.validate()?;
        Ok(authority)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.plan.validate().map_err(invalid)?;
        if !self.topology.has_valid_identity()
            || !nonempty_identity(self.workload.digest())
            || !nonempty_identity(self.result.digest())
            || !nonempty_identity(self.producer.digest())
            || self.entries.len() != self.plan.frequencies_hz().len()
        {
            return Err(invalid(
                "frequency authority requires complete plan and topology",
            ));
        }
        let mut sources = BTreeSet::new();
        let mut workloads = BTreeSet::new();
        for (expected, actual) in self
            .plan
            .entries()
            .map_err(invalid)?
            .iter()
            .zip(&self.entries)
        {
            let source_digest = match actual.source {
                FieldRecoverySource::SolverResult(result) => result.digest(),
                FieldRecoverySource::StructuralFrequencyEntry(result) => result.digest(),
            };
            if actual.entry != *expected
                || !nonempty_identity(actual.workload.digest())
                || !nonempty_identity(source_digest)
                || !sources.insert(actual.source)
                || !workloads.insert(actual.workload)
            {
                return Err(AnalysisExecutionError::Identity(
                    "frequency numerical inventory is missing, duplicate, foreign or unordered",
                ));
            }
        }
        Ok(())
    }

    pub const fn workload(&self) -> AnalysisWorkloadIdentity {
        self.workload
    }
    pub const fn result(&self) -> SolverResultIdentity {
        self.result
    }
    pub const fn producer(&self) -> StepIdentity {
        self.producer
    }
    pub const fn topology(&self) -> FieldTopologyIdentity {
        self.topology
    }
    pub fn plan(&self) -> &FrequencyResponsePlan {
        &self.plan
    }
    pub fn entries(&self) -> &[FrequencySampleNumericalEntry] {
        &self.entries
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrequencySamplePublicationSample {
    pub entry: FrequencyResponsePlanEntry,
    pub recovery_request: FieldRecoveryRequest,
    pub fields: RecoveredFieldArtifactManifest,
    pub observations: Option<ObservationArtifactManifest>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ManifestWire")]
pub struct FrequencySamplePublicationManifest {
    schema_version: u16,
    identity: FrequencySamplePublicationIdentity,
    authority: FrequencySamplePublicationAuthority,
    samples: Vec<FrequencySamplePublicationSample>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema_version: u16,
    identity: FrequencySamplePublicationIdentity,
    authority: FrequencySamplePublicationAuthority,
    samples: Vec<FrequencySamplePublicationSample>,
}

impl TryFrom<ManifestWire> for FrequencySamplePublicationManifest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            authority: wire.authority,
            samples: wire.samples,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl FrequencySamplePublicationManifest {
    pub fn new(
        authority: FrequencySamplePublicationAuthority,
        samples: Vec<FrequencySamplePublicationSample>,
    ) -> AnalysisExecutionResult<Self> {
        let mut manifest = Self {
            schema_version: 1,
            identity: FrequencySamplePublicationIdentity(Digest::sha256([])),
            authority,
            samples,
        };
        manifest.identity = manifest.derive_identity()?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub const fn identity(&self) -> FrequencySamplePublicationIdentity {
        self.identity
    }
    pub fn authority(&self) -> &FrequencySamplePublicationAuthority {
        &self.authority
    }
    pub fn samples(&self) -> &[FrequencySamplePublicationSample] {
        &self.samples
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.authority.validate()?;
        if self.schema_version != 1 || self.samples.len() != self.authority.entries.len() {
            return Err(invalid(
                "frequency publication must cover every planned sample",
            ));
        }
        let mut field_samples = BTreeSet::new();
        for (sample, numerical) in self.samples.iter().zip(&self.authority.entries) {
            sample.recovery_request.validate()?;
            sample.fields.validate().map_err(invalid)?;
            let request = &sample.recovery_request;
            let coordinate = RecoveredFieldSample::FrequencyHertz(numerical.entry.frequency_hz());
            if sample.entry != numerical.entry
                || request.producer != self.authority.producer
                || request.topology != self.authority.topology
                || request.source != numerical.source
                || sample.fields.recovery_request != request.identity()
                || sample.fields.source != numerical.source
                || sample.fields.fields.len() != request.plan.requirements.len()
                || sample.fields.topology_digest != self.authority.topology.discretization_digest()
                || sample
                    .fields
                    .fields
                    .iter()
                    .any(|field| field.sample != coordinate)
            {
                return Err(AnalysisExecutionError::Identity(
                    "frequency sample catalog differs from exact entry, producer, source or topology",
                ));
            }
            request.plan.validate_definitions(
                sample
                    .fields
                    .fields
                    .iter()
                    .map(|field| field.definition.clone()),
            )?;
            for field in &sample.fields.fields {
                if !field_samples.insert((
                    field.definition.field_id(),
                    numerical.entry.frequency_hz().to_bits(),
                )) {
                    return Err(invalid("duplicate semantic field and frequency sample"));
                }
            }
            if let Some(observations) = &sample.observations {
                observations.validate().map_err(invalid)?;
                validate_observations(
                    sample,
                    observations,
                    self.authority.producer,
                    self.authority.topology,
                )?;
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "frequency publication identity does not converge",
            ));
        }
        Ok(())
    }

    /// Internal integrity is not external numerical authority: every consumer must additionally
    /// compare with the independently authenticated aggregate/entry inventory.
    pub fn validate_for(
        &self,
        authority: &FrequencySamplePublicationAuthority,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        authority.validate()?;
        if &self.authority != authority {
            return Err(AnalysisExecutionError::Identity(
                "frequency publication names foreign numerical authority",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "frequency-sample-publication",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "frequency-sample-publication",
            bytes,
            LIMITS,
        )?)
    }

    /// Exact scientific selection. Omission is allowed only for a singleton inventory.
    pub fn select_sample(
        &self,
        selection: Option<RecoveredFieldSample>,
    ) -> AnalysisExecutionResult<&FrequencySamplePublicationSample> {
        self.validate()?;
        let frequency = match selection {
            Some(RecoveredFieldSample::FrequencyHertz(frequency))
                if frequency.is_finite() && frequency > 0.0 =>
            {
                frequency
            }
            None if self.samples.len() == 1 => self.samples[0].entry.frequency_hz(),
            None => {
                return Err(invalid(
                    "frequency sample selection is required for a multi-sample publication",
                ))
            }
            _ => {
                return Err(invalid(
                    "frequency selection requires a positive finite frequency coordinate",
                ))
            }
        };
        self.samples
            .iter()
            .find(|sample| sample.entry.frequency_hz().to_bits() == frequency.to_bits())
            .ok_or_else(|| invalid("selected frequency is absent from the complete publication"))
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<FrequencySamplePublicationIdentity> {
        Ok(FrequencySamplePublicationIdentity(Digest::sha256(
            encode_contract(
                PREFIX,
                "frequency-sample-publication-identity",
                &(self.schema_version, &self.authority, &self.samples),
                LIMITS,
            )?,
        )))
    }
}

fn validate_observations(
    sample: &FrequencySamplePublicationSample,
    observations: &ObservationArtifactManifest,
    producer: StepIdentity,
    topology: FieldTopologyIdentity,
) -> AnalysisExecutionResult<()> {
    let source_matches = match observations.source {
        ObservationFieldSource::SampledRecovery {
            recovery_request,
            artifact,
            numerical_source,
            topology: source_topology,
        } => {
            recovery_request == sample.recovery_request.identity()
                && artifact == sample.fields.identity
                && numerical_source == sample.fields.source
                && source_topology == topology
        }
        ObservationFieldSource::StructuralRecovery {
            recovery_request,
            artifact,
            numerical_source,
            topology: source_topology,
        } => {
            recovery_request == sample.recovery_request.identity()
                && artifact == sample.fields.identity
                && numerical_source == sample.fields.source
                && topology
                    == FieldTopologyIdentity::Structural {
                        topology: source_topology,
                    }
        }
        ObservationFieldSource::VolumeMeshRecovery {
            recovery_request,
            artifact,
            numerical_source,
            topology: source_topology,
            ..
        } => {
            recovery_request == sample.recovery_request.identity()
                && artifact == sample.fields.identity
                && numerical_source == sample.fields.source
                && source_topology == topology
        }
        ObservationFieldSource::ThermalResult { .. } => false,
    };
    // Volume-mesh plans address the mesh's analysis projection, named by the source.
    let observation_topology = match observations.source {
        ObservationFieldSource::VolumeMeshRecovery { .. } => {
            observations.source.observation_topology_digest()
        }
        _ => topology.discretization_digest(),
    };
    let coordinate = RecoveredFieldSample::FrequencyHertz(sample.entry.frequency_hz());
    if !source_matches
        || observations.producer != producer
        || observations.topology_digest != observation_topology
        || observations
            .results
            .iter()
            .any(|result| result.sample != coordinate)
    {
        return Err(AnalysisExecutionError::Identity(
            "frequency observations differ from their exact field sample",
        ));
    }
    Ok(())
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}

fn nonempty_identity(digest: Digest) -> bool {
    digest != Digest::from_bytes([0; 32]) && digest != Digest::sha256([])
}

#[cfg(test)]
mod tests;
