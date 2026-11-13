use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::execution::{
    AccuracyEvaluationRequestIdentity, AnalysisExecutionError, AnalysisExecutionResult,
    PublicationIdentity, PublicationRequestIdentity, ResultPublicationRequest,
    SolverResultIdentity, StepIdentity,
};
use crate::planning::resource::{
    FeaResourcePlanIdentity, ResourceUsageEvidence, ResourceUsageEvidenceIdentity,
};

use super::accuracy::{
    AccuracyArtifact, AccuracyArtifactIdentity, AccuracyPublicationGate, AccuracyStatus,
    PublicationDecision,
};
use super::observation::ObservationPublicationCatalogIdentity;

// Hard cutover: receipts must carry the independently bound accuracy task authority.
const PUBLICATION_PREFIX: &[u8] = b"runmat-analysis-published-result/v3\0";
const PUBLICATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(8 * 1024 * 1024, 1_000_000, 65_536, 64);

mod observation_usage;
mod profile;
mod usage;
pub(crate) use observation_usage::observation_usage_index_bytes;
pub use observation_usage::validate_observation_publication_usage;
pub use profile::published_result_encoding_profile;
pub(crate) use usage::validate_accuracy_resource_usage;
pub use usage::{
    validate_result_publication_resource_usage, validate_retained_result_publication_resource_usage,
};

/// Immutable terminal receipt for a result set admitted by its typed accuracy policy.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, try_from = "PublishedAnalysisResultWire")]
pub struct PublishedAnalysisResult {
    pub schema_version: u16,
    identity: PublicationIdentity,
    pub request: PublicationRequestIdentity,
    pub producer: StepIdentity,
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationPublicationCatalogIdentity>,
    pub accuracy_request: AccuracyEvaluationRequestIdentity,
    pub source_accuracy: AccuracyArtifactIdentity,
    pub source_resource_plan: FeaResourcePlanIdentity,
    pub source_resource_usage: ResourceUsageEvidenceIdentity,
    pub accuracy_status: AccuracyStatus,
    pub gate: AccuracyPublicationGate,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishedAnalysisResultWire {
    schema_version: u16,
    identity: PublicationIdentity,
    request: PublicationRequestIdentity,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationPublicationCatalogIdentity>,
    accuracy_request: AccuracyEvaluationRequestIdentity,
    source_accuracy: AccuracyArtifactIdentity,
    source_resource_plan: FeaResourcePlanIdentity,
    source_resource_usage: ResourceUsageEvidenceIdentity,
    accuracy_status: AccuracyStatus,
    gate: AccuracyPublicationGate,
}

impl TryFrom<PublishedAnalysisResultWire> for PublishedAnalysisResult {
    type Error = AnalysisExecutionError;

    fn try_from(wire: PublishedAnalysisResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            producer: wire.producer,
            source_result: wire.source_result,
            source_observations: wire.source_observations,
            accuracy_request: wire.accuracy_request,
            source_accuracy: wire.source_accuracy,
            source_resource_plan: wire.source_resource_plan,
            source_resource_usage: wire.source_resource_usage,
            accuracy_status: wire.accuracy_status,
            gate: wire.gate,
        };
        result.validate()?;
        Ok(result)
    }
}

impl PublishedAnalysisResult {
    pub fn new(
        request: &ResultPublicationRequest,
        accuracy: &AccuracyArtifact,
        resource_usage: &ResourceUsageEvidence,
    ) -> AnalysisExecutionResult<Self> {
        profile::validate_sources(request, accuracy, resource_usage)?;
        let identity = PublicationIdentity::new(
            request.identity(),
            request.source_result,
            request
                .source_observations
                .map(|identity| identity.digest()),
            request.source_accuracy.digest(),
            request.source_resource_plan.digest(),
            request.source_resource_usage.digest(),
        )?;
        let result = Self {
            schema_version: 3,
            identity,
            request: request.identity(),
            producer: request.producer,
            source_result: request.source_result,
            source_observations: request.source_observations,
            accuracy_request: request.accuracy_request,
            source_accuracy: request.source_accuracy,
            source_resource_plan: request.source_resource_plan,
            source_resource_usage: request.source_resource_usage,
            accuracy_status: accuracy.evaluation().report().status,
            gate: accuracy.gate().clone(),
        };
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> PublicationIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        ResultPublicationRequest::from_publication(self)?;
        let expected = PublicationIdentity::new(
            self.request,
            self.source_result,
            self.source_observations.map(|identity| identity.digest()),
            self.source_accuracy.digest(),
            self.source_resource_plan.digest(),
            self.source_resource_usage.digest(),
        )?;
        if self.schema_version != 3
            || self.identity != expected
            || self.gate.decision != PublicationDecision::Allow
            || self.gate.status != self.accuracy_status
        {
            return Err(AnalysisExecutionError::Invalid(
                "published result lacks a convergent identity or an allowing accuracy gate".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_sources(
        &self,
        accuracy: &AccuracyArtifact,
        resource_usage: &ResourceUsageEvidence,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        accuracy
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        validate_accuracy_resource_usage(
            self.accuracy_request,
            self.source_result,
            resource_usage,
        )?;
        let resource_plan = resource_usage
            .primary_plan()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if self.producer != accuracy.producer
            || self.accuracy_status != accuracy.evaluation().report().status
            || &self.gate != accuracy.gate()
            || self.source_result != accuracy.source_result
            || self.source_observations != accuracy.source_observations
            || self.accuracy_request != accuracy.evaluation_request
            || self.source_accuracy != accuracy.identity()
            || self.source_resource_plan != resource_plan.identity
            || self.source_resource_usage != resource_usage.identity()
            || resource_usage.primary_plan != resource_plan.identity
            || resource_usage.result_digest != self.source_result.digest()
            || !resource_usage.overruns.is_empty()
        {
            return Err(AnalysisExecutionError::Identity(
                "published result differs from its governed accuracy or resource sources",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PUBLICATION_PREFIX,
            "published-analysis-result",
            self,
            PUBLICATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let result = decode_contract(
            PUBLICATION_PREFIX,
            "published-analysis-result",
            bytes,
            PUBLICATION_LIMITS,
        )?;
        Ok(result)
    }
}
