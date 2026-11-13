use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::results::observation::{
    ObservationPartitionEvidenceIdentity, ObservationResult, ObservationResultIdentity,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ObservationAssemblyRequestIdentity,
    ObservationEvaluationRequest, ObservationEvaluationRequestIdentity, StepIdentity,
};

const ASSEMBLY_PREFIX: &[u8] = b"runmat-analysis-observation-assembly/v1\0";
const ASSEMBLY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 32));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationAssemblyRequestWire")]
pub struct ObservationAssemblyRequest {
    pub schema_version: u16,
    identity: ObservationAssemblyRequestIdentity,
    pub producer: StepIdentity,
    pub source_request: ObservationEvaluationRequestIdentity,
    pub results: Vec<ObservationResultIdentity>,
    pub partition_evidence: Vec<ObservationPartitionEvidenceIdentity>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationAssemblyRequestWire {
    schema_version: u16,
    identity: ObservationAssemblyRequestIdentity,
    producer: StepIdentity,
    source_request: ObservationEvaluationRequestIdentity,
    results: Vec<ObservationResultIdentity>,
    partition_evidence: Vec<ObservationPartitionEvidenceIdentity>,
}

impl TryFrom<ObservationAssemblyRequestWire> for ObservationAssemblyRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ObservationAssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_request: wire.source_request,
            results: wire.results,
            partition_evidence: wire.partition_evidence,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl ObservationAssemblyRequest {
    /// Reconstruct the exact assembly request without loading numerical result payloads.
    /// The publication catalog independently validates scheduled result coverage.
    pub(crate) fn published_identity(
        source: &ObservationEvaluationRequest,
        manifest: &crate::results::observation::ObservationArtifactManifest,
    ) -> AnalysisExecutionResult<ObservationAssemblyRequestIdentity> {
        source.validate()?;
        manifest
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if manifest.evaluation_request != source.identity()
            || manifest.producer != source.producer
            || manifest.plan_identity != source.plan.identity
            || manifest.source_selection != source.selection.identity()
            || manifest.source != source.selection.source()
        {
            return Err(AnalysisExecutionError::Identity(
                "published assembly differs from its source request",
            ));
        }
        assembly_identity(
            source.producer,
            source.identity(),
            &ManifestIdentities::Results(&manifest.results),
            &ManifestIdentities::Evidence(&manifest.results),
        )
    }

    pub fn new(
        source: &ObservationEvaluationRequest,
        results: &[ObservationResult],
    ) -> AnalysisExecutionResult<Self> {
        validate_results(source, results)?;
        let result_identities = results
            .iter()
            .map(|result| result.logical_identity)
            .collect::<Vec<_>>();
        let evidence_identities = results
            .iter()
            .map(|result| result.partition_evidence.identity)
            .collect::<Vec<_>>();
        Self::from_identities(source, result_identities, evidence_identities)
    }

    fn from_identities(
        source: &ObservationEvaluationRequest,
        result_identities: Vec<ObservationResultIdentity>,
        evidence_identities: Vec<ObservationPartitionEvidenceIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
            identity: assembly_identity(
                source.producer,
                source.identity(),
                &result_identities,
                &evidence_identities,
            )?,
            producer: source.producer,
            source_request: source.identity(),
            results: result_identities,
            partition_evidence: evidence_identities,
        };
        request.validate_identity()?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationAssemblyRequestIdentity {
        self.identity
    }

    pub fn validate_for(
        &self,
        source: &ObservationEvaluationRequest,
        results: &[ObservationResult],
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        validate_results(source, results)?;
        if self.producer != source.producer
            || self.source_request != source.identity()
            || self.results
                != results
                    .iter()
                    .map(|result| result.logical_identity)
                    .collect::<Vec<_>>()
            || self.partition_evidence
                != results
                    .iter()
                    .map(|result| result.partition_evidence.identity)
                    .collect::<Vec<_>>()
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly inputs differ from its request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            ASSEMBLY_PREFIX,
            "observation-assembly-request",
            self,
            ASSEMBLY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            ASSEMBLY_PREFIX,
            "observation-assembly-request",
            bytes,
            ASSEMBLY_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        let expected = assembly_identity(
            self.producer,
            self.source_request,
            &self.results,
            &self.partition_evidence,
        )?;
        if self.schema_version != 1
            || self.results.is_empty()
            || self.results.len() != self.partition_evidence.len()
            || self.identity != expected
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly request identity does not converge",
            ));
        }
        Ok(())
    }
}

fn assembly_identity(
    producer: StepIdentity,
    source: ObservationEvaluationRequestIdentity,
    results: &impl Serialize,
    evidence: &impl Serialize,
) -> AnalysisExecutionResult<ObservationAssemblyRequestIdentity> {
    ObservationAssemblyRequestIdentity::derive(
        "observation-assembly-request",
        &(producer, source, results, evidence),
    )
}

enum ManifestIdentities<'a> {
    Results(&'a [crate::results::observation::ObservationArtifactEntry]),
    Evidence(&'a [crate::results::observation::ObservationArtifactEntry]),
}
impl Serialize for ManifestIdentities<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let entries = match self {
            Self::Results(entries) | Self::Evidence(entries) => entries,
        };
        let mut sequence = serializer.serialize_seq(Some(entries.len()))?;
        for entry in *entries {
            match self {
                Self::Results(_) => sequence.serialize_element(&entry.result)?,
                Self::Evidence(_) => sequence.serialize_element(&entry.partition_evidence)?,
            }
        }
        sequence.end()
    }
}

fn validate_results(
    source: &ObservationEvaluationRequest,
    results: &[ObservationResult],
) -> AnalysisExecutionResult<()> {
    source.validate()?;
    let first = results.first().ok_or(AnalysisExecutionError::Invalid(
        "observation assembly requires at least one scheduled result".into(),
    ))?;
    let sample = first.sample;
    let event_attribution = first.partition_evidence.event_attribution;
    if results.iter().any(|result| {
        result.sample != sample || result.partition_evidence.event_attribution != event_attribution
    }) {
        return Err(AnalysisExecutionError::Identity(
            "observation assembly mixes sample coordinates or event attribution",
        ));
    }
    let expected = source.scheduled_observations(sample, event_attribution)?;
    if expected.len() != results.len() {
        return Err(AnalysisExecutionError::Identity(
            "observation assembly does not exactly cover the scheduled sample",
        ));
    }
    for (observation, result) in expected.into_iter().zip(results) {
        result
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if result.plan_identity != source.plan.identity
            || result.topology_digest != source.plan.topology_digest
            || result.observation_id != observation.spec.observation_id
        {
            return Err(AnalysisExecutionError::Identity(
                "observation assembly result order or source differs from its resolved plan",
            ));
        }
    }
    Ok(())
}
