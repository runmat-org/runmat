//! Borrowed receipt serialization: profile and actual wire output share every field.
use super::*;
use runmat_canonical_codec::{profile_serialized, SerializedCanonicalProfile};

#[derive(Serialize)]
struct Receipt<'a> {
    schema_version: u16,
    identity: PublicationIdentity,
    request: PublicationRequestIdentity,
    producer: StepIdentity,
    source_result: SolverResultIdentity,
    source_observations: Option<ObservationPublicationCatalogIdentity>,
    source_accuracy: AccuracyArtifactIdentity,
    source_resource_plan: FeaResourcePlanIdentity,
    source_resource_usage: ResourceUsageEvidenceIdentity,
    formulation_requirement: FormulationRequirement,
    accuracy_status: AccuracyStatus,
    gate: &'a AccuracyPublicationGate,
}

impl Serialize for PublishedAnalysisResult {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Receipt {
            schema_version: self.schema_version,
            identity: self.identity,
            request: self.request,
            producer: self.producer,
            source_result: self.source_result,
            source_observations: self.source_observations,
            source_accuracy: self.source_accuracy,
            source_resource_plan: self.source_resource_plan,
            source_resource_usage: self.source_resource_usage,
            formulation_requirement: self.formulation_requirement,
            accuracy_status: self.accuracy_status,
            gate: &self.gate,
        }
        .serialize(serializer)
    }
}

pub(super) fn validate_sources(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<()> {
    request.validate()?;
    accuracy
        .validate()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let primary = usage
        .primary_plan()
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    if request.producer != accuracy.producer
        || request.source_result != accuracy.source_result
        || request.source_observations != accuracy.source_observations
        || request.accuracy_request != accuracy.evaluation_request
        || request.source_accuracy != accuracy.identity()
        || request.source_resource_plan != primary.identity
        || request.source_resource_usage != usage.identity()
        || usage.result_digest != request.source_result.digest()
        || !usage.overruns.is_empty()
        || accuracy.gate().decision != PublicationDecision::Allow
        || (accuracy.formulation_requirement == FormulationRequirement::Production
            && accuracy.evaluation().report().status != AccuracyStatus::Satisfied)
    {
        return Err(AnalysisExecutionError::Identity(
            "published result sources differ from the admitted request or publication gate",
        ));
    }
    Ok(())
}

/// Profiles the exact governed receipt without constructing or publishing one on the driver.
pub fn published_result_encoding_profile(
    request: &ResultPublicationRequest,
    accuracy: &AccuracyArtifact,
    usage: &ResourceUsageEvidence,
) -> AnalysisExecutionResult<SerializedCanonicalProfile> {
    validate_sources(request, accuracy, usage)?;
    let identity = PublicationIdentity::new(
        request.identity(),
        request.source_result,
        request.source_observations.map(|id| id.digest()),
        request.source_accuracy.digest(),
        request.source_resource_plan.digest(),
        request.source_resource_usage.digest(),
    )?;
    Ok(profile_serialized(
        &Receipt {
            schema_version: 2,
            identity,
            request: request.identity(),
            producer: request.producer,
            source_result: request.source_result,
            source_observations: request.source_observations,
            source_accuracy: request.source_accuracy,
            source_resource_plan: request.source_resource_plan,
            source_resource_usage: request.source_resource_usage,
            formulation_requirement: accuracy.formulation_requirement,
            accuracy_status: accuracy.evaluation().report().status,
            gate: accuracy.gate(),
        },
        PUBLICATION_LIMITS,
    )?)
}
