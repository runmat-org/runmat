use runmat_analysis_core::{ObservationValueShape, PartialAccessPolicy};

use crate::execution::{
    chunk_numerical_payload, AnalysisNumericalObjectKind, ChunkedAnalysisPayload,
};

use super::contract::{
    publication_result, ObservationEntity, ObservationError, ObservationResult,
    ResolvedObservationPlan, ResolvedObservationTarget,
};

pub fn publish_observation(
    plan: &ResolvedObservationPlan,
    result: &ObservationResult,
    requested_chunk_bytes: usize,
) -> Result<ChunkedAnalysisPayload, ObservationError> {
    plan.validate_identity()?;
    let planned = plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == result.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("observation result is absent from its resolved plan".into())
        })?;
    let expected_shape = if planned.result_component_count == 1 {
        ObservationValueShape::Scalar
    } else {
        planned.spec.value_shape.clone()
    };
    let expected_entities = match &planned.target {
        ResolvedObservationTarget::Nodes { identities, .. } => identities
            .iter()
            .copied()
            .map(ObservationEntity::Node)
            .collect::<Vec<_>>(),
        ResolvedObservationTarget::Elements { identities, .. } => identities
            .iter()
            .copied()
            .map(ObservationEntity::Element)
            .collect(),
        ResolvedObservationTarget::Volume { .. } | ResolvedObservationTarget::Boundary { .. } => {
            Vec::new()
        }
    };
    if result.plan_identity != plan.identity
        || result.topology_digest != plan.topology_digest
        || result.partial_access != planned.spec.storage.partial_access
        || result.quantity != planned.spec.quantity
        || result.location != planned.spec.output_location
        || result.value_shape != expected_shape
        || result.entities != expected_entities
    {
        return Err(ObservationError::Identity(
            "observation result differs from its resolved plan",
        ));
    }
    let bytes = result.canonical_bytes()?;
    let chunk_bytes = match result.partial_access {
        PartialAccessPolicy::ManifestAndChunks => requested_chunk_bytes,
        PartialAccessPolicy::WholeArtifact => bytes.len(),
    };
    publication_result(chunk_numerical_payload(
        AnalysisNumericalObjectKind::Observation,
        result.logical_identity.digest(),
        &bytes,
        chunk_bytes,
    ))
}
