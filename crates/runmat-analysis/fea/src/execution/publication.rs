use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::planning::resource::{
    FeaResourcePlanIdentity, ResourceUsageEvidence, ResourceUsageEvidenceIdentity,
};
use crate::results::accuracy::{AccuracyArtifact, AccuracyArtifactIdentity};
use crate::results::observation::{
    ObservationPublicationCatalog, ObservationPublicationCatalogIdentity,
};

use super::{
    AccuracyEvaluationRequestIdentity, AnalysisExecutionError, AnalysisExecutionResult,
    PublicationRequestIdentity, SolverResultIdentity, StepIdentity,
};

const PUBLICATION_REQUEST_PREFIX: &[u8] = b"runmat-analysis-publication-request/v2\0";
const PUBLICATION_REQUEST_LIMITS: CanonicalLimits =
    CanonicalLimits::new(8 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ResultPublicationRequestWire")]
pub struct ResultPublicationRequest {
    pub schema_version: u16,
    identity: PublicationRequestIdentity,
    pub producer: StepIdentity,
    pub source_result: SolverResultIdentity,
    pub source_observations: Option<ObservationPublicationCatalogIdentity>,
    pub accuracy_request: AccuracyEvaluationRequestIdentity,
    pub source_accuracy: AccuracyArtifactIdentity,
    pub source_resource_plan: FeaResourcePlanIdentity,
    pub source_resource_usage: ResourceUsageEvidenceIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultPublicationRequestWire {
    schema_version: u16,
    identity: PublicationRequestIdentity,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationPublicationCatalogIdentity>,
    accuracy_request: AccuracyEvaluationRequestIdentity,
    source_accuracy: AccuracyArtifactIdentity,
    source_resource_plan: FeaResourcePlanIdentity,
    source_resource_usage: ResourceUsageEvidenceIdentity,
}

impl TryFrom<ResultPublicationRequestWire> for ResultPublicationRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ResultPublicationRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source_result: wire.source_result,
            source_observations: wire.source_observations,
            accuracy_request: wire.accuracy_request,
            source_accuracy: wire.source_accuracy,
            source_resource_plan: wire.source_resource_plan,
            source_resource_usage: wire.source_resource_usage,
        };
        request.validate()?;
        Ok(request)
    }
}

impl ResultPublicationRequest {
    pub fn new(
        accuracy: &AccuracyArtifact,
        observations: Option<&ObservationPublicationCatalog>,
        resource_usage: &ResourceUsageEvidence,
    ) -> AnalysisExecutionResult<Self> {
        accuracy
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        crate::results::publication::validate_accuracy_resource_usage(
            accuracy.evaluation_request,
            accuracy.source_result,
            resource_usage,
        )?;
        let observation_identity = match observations {
            Some(observations) => {
                crate::results::publication::validate_observation_publication_usage(
                    observations,
                    resource_usage,
                )?;
                observations.validate_source_result(
                    accuracy.producer,
                    accuracy.source_kind,
                    accuracy.source_result,
                )?;
                if Some(observations.identity()) != accuracy.source_observations {
                    return Err(AnalysisExecutionError::Identity(
                        "publication observations differ from the accuracy source",
                    ));
                }
                Some(observations.identity())
            }
            None if accuracy.source_observations.is_none() => None,
            None => {
                return Err(AnalysisExecutionError::Identity(
                    "publication omits observations evaluated by its accuracy source",
                ))
            }
        };
        let resource_plan = resource_usage
            .primary_plan()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        if resource_usage.primary_plan != resource_plan.identity
            || resource_usage.result_digest != accuracy.source_result.digest()
            || !resource_usage.overruns.is_empty()
        {
            return Err(AnalysisExecutionError::Identity(
                "publication resource evidence differs from the accuracy source",
            ));
        }
        let identity = PublicationRequestIdentity::derive(
            "result-publication-request",
            &(
                accuracy.producer,
                accuracy.source_result,
                observation_identity,
                accuracy.evaluation_request,
                accuracy.identity(),
                resource_plan.identity,
                resource_usage.identity(),
            ),
        )?;
        let request = Self {
            schema_version: 2,
            identity,
            producer: accuracy.producer,
            source_result: accuracy.source_result,
            source_observations: observation_identity,
            accuracy_request: accuracy.evaluation_request,
            source_accuracy: accuracy.identity(),
            source_resource_plan: resource_plan.identity,
            source_resource_usage: resource_usage.identity(),
        };
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> PublicationRequestIdentity {
        self.identity
    }

    /// Reconstruct the original request from a receipt, including its accuracy task authority.
    /// This prevents a receipt from substituting the expected task while retaining its request ID.
    pub(crate) fn from_publication(
        publication: &crate::results::publication::PublishedAnalysisResult,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 2,
            identity: publication.request,
            producer: publication.producer,
            source_result: publication.source_result,
            source_observations: publication.source_observations,
            accuracy_request: publication.accuracy_request,
            source_accuracy: publication.source_accuracy,
            source_resource_plan: publication.source_resource_plan,
            source_resource_usage: publication.source_resource_usage,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let expected = PublicationRequestIdentity::derive(
            "result-publication-request",
            &(
                self.producer,
                self.source_result,
                self.source_observations,
                self.accuracy_request,
                self.source_accuracy,
                self.source_resource_plan,
                self.source_resource_usage,
            ),
        )?;
        if self.schema_version != 2 || self.identity != expected {
            return Err(AnalysisExecutionError::Identity(
                "result publication request identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PUBLICATION_REQUEST_PREFIX,
            "result-publication-request",
            self,
            PUBLICATION_REQUEST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request = decode_contract(
            PUBLICATION_REQUEST_PREFIX,
            "result-publication-request",
            bytes,
            PUBLICATION_REQUEST_LIMITS,
        )?;
        Ok(request)
    }
}
