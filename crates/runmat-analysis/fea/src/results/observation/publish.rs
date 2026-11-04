use runmat_analysis_core::{ExtremaTiePolicy, PartialAccessPolicy, SpatialReduction};

use crate::execution::{
    chunk_numerical_payload, AnalysisNumericalObjectKind, ChunkedAnalysisPayload,
};

use super::contract::{
    observation_element_entity, publication_result, ObservationEntity, ObservationError,
    ObservationResult, ResolvedObservationPlan, ResolvedObservationTarget,
};

pub fn publish_observation(
    plan: &ResolvedObservationPlan,
    result: &ObservationResult,
    requested_chunk_bytes: usize,
) -> Result<ChunkedAnalysisPayload, ObservationError> {
    let planned = plan
        .observations
        .iter()
        .find(|observation| observation.spec.observation_id == result.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("observation result is absent from its resolved plan".into())
        })?;
    let expected_shape = &planned.result_value_shape;
    let expected_entities = match &planned.target {
        ResolvedObservationTarget::Nodes { identities, .. } => identities
            .iter()
            .copied()
            .map(ObservationEntity::Node)
            .collect::<Vec<_>>(),
        ResolvedObservationTarget::Elements {
            domain, identities, ..
        } => identities
            .iter()
            .copied()
            .map(|identity| observation_element_entity(*domain, identity))
            .collect(),
        ResolvedObservationTarget::Points { samples } => samples
            .iter()
            .map(|sample| ObservationEntity::Point {
                identity: sample.identity,
                ordinal: sample.ordinal,
                position_m: sample.position_m,
                source_element: sample.source_element_identity,
            })
            .collect(),
        ResolvedObservationTarget::Connectors { identities, .. } => identities
            .iter()
            .copied()
            .map(ObservationEntity::Connector)
            .collect(),
        ResolvedObservationTarget::Volume { .. }
        | ResolvedObservationTarget::Boundary { .. }
        | ResolvedObservationTarget::BoundaryPair { .. }
        | ResolvedObservationTarget::ConstraintGroups { .. } => Vec::new(),
    };
    let spatial_extrema = planned.spec.reduction.as_ref().is_some_and(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
    });
    let entities_match = if spatial_extrema {
        let first_only =
            planned.spec.reduction.as_ref().is_some_and(|reduction| {
                reduction.extrema_ties == ExtremaTiePolicy::FirstCanonical
            });
        !result.entities.is_empty()
            && (!first_only || result.entities.len() == 1)
            && result.entities.windows(2).all(|pair| {
                let left = expected_entities
                    .iter()
                    .position(|entity| entity == &pair[0]);
                let right = expected_entities
                    .iter()
                    .position(|entity| entity == &pair[1]);
                left.zip(right).is_some_and(|(left, right)| left < right)
            })
            && result
                .entities
                .iter()
                .all(|entity| expected_entities.contains(entity))
    } else {
        result.entities == expected_entities
    };
    if result.plan_identity != plan.identity
        || result.topology_digest != plan.topology_digest
        || result.partial_access != planned.spec.storage.partial_access
        || result.quantity != planned.spec.quantity
        || result.location != planned.spec.output_location
        || &result.value_shape != expected_shape
        || result.complex_representation != planned.spec.complex_representation
        || !entities_match
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
