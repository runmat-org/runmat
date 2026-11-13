use runmat_analysis_core::{ExtremaTiePolicy, SpatialReduction};

use super::{
    ObservationEntity, ObservationError, ObservationPartition, ObservationPartitionPayload,
    ObservationPartitionTarget, ResolvedObservation, ResolvedObservationTarget,
};

pub(super) fn selected_or_extrema_payload(
    observation: &ResolvedObservation,
    entities: Vec<ObservationEntity>,
    values: Vec<f64>,
) -> Result<ObservationPartitionPayload, ObservationError> {
    if spatial_extrema(observation).is_some() {
        let (entities, values) = extrema_values(observation, entities, values)?;
        Ok(ObservationPartitionPayload::ExtremaCandidates { entities, values })
    } else {
        Ok(ObservationPartitionPayload::SelectedValues { entities, values })
    }
}

pub(super) fn spatial_extrema(observation: &ResolvedObservation) -> Option<SpatialReduction> {
    observation.spec.reduction.as_ref().and_then(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
        .then_some(reduction.spatial)
    })
}

pub(super) fn extrema_values(
    observation: &ResolvedObservation,
    entities: Vec<ObservationEntity>,
    values: Vec<f64>,
) -> Result<(Vec<ObservationEntity>, Vec<f64>), ObservationError> {
    if entities.is_empty() || entities.len() != values.len() {
        return Err(ObservationError::Identity(
            "spatial extrema candidates must contain one scalar per entity",
        ));
    }
    let operation = spatial_extrema(observation).ok_or(ObservationError::Identity(
        "extrema candidates require a spatial extrema policy",
    ))?;
    let extreme = values
        .iter()
        .copied()
        .reduce(|left, right| match operation {
            SpatialReduction::Minimum => left.min(right),
            SpatialReduction::Maximum => left.max(right),
            _ => unreachable!(),
        })
        .ok_or_else(|| ObservationError::Invalid("spatial extrema candidates are empty".into()))?;
    let ties = observation
        .spec
        .reduction
        .as_ref()
        .map(|reduction| reduction.extrema_ties)
        .ok_or(ObservationError::Identity(
            "spatial extrema result lacks a tie policy",
        ))?;
    let mut selected_entities = Vec::new();
    for (entity, value) in entities.into_iter().zip(values) {
        if value == extreme {
            selected_entities.push(entity);
            if ties == ExtremaTiePolicy::FirstCanonical {
                break;
            }
        }
    }
    let selected_values = vec![extreme; selected_entities.len()];
    Ok((selected_entities, selected_values))
}

pub(super) fn partition_selected_entities(
    partition: &ObservationPartition,
    observation: &ResolvedObservation,
) -> Result<Vec<ObservationEntity>, ObservationError> {
    match &partition.target {
        ObservationPartitionTarget::Nodes { identities, .. } => Ok(identities
            .iter()
            .copied()
            .map(ObservationEntity::Node)
            .collect()),
        ObservationPartitionTarget::Elements {
            domain, identities, ..
        } => Ok(identities
            .iter()
            .copied()
            .map(|identity| super::contract::observation_element_entity(*domain, identity))
            .collect()),
        ObservationPartitionTarget::Points {
            sample_ordinals, ..
        } => {
            let ResolvedObservationTarget::Points { samples } = &observation.target else {
                return Err(ObservationError::Identity(
                    "point partition does not match its resolved target",
                ));
            };
            sample_ordinals
                .iter()
                .map(|ordinal| {
                    let sample = samples.get(*ordinal).ok_or(ObservationError::Identity(
                        "point partition exceeds its resolved target",
                    ))?;
                    Ok(ObservationEntity::Point {
                        identity: sample.identity,
                        ordinal: sample.ordinal,
                        position_m: sample.position_m,
                        source_element: sample.source_element_identity,
                    })
                })
                .collect()
        }
        ObservationPartitionTarget::Volume { .. }
        | ObservationPartitionTarget::Boundary { .. }
        | ObservationPartitionTarget::BoundaryPair { .. }
        | ObservationPartitionTarget::Connectors { .. }
        | ObservationPartitionTarget::ConstraintGroups { .. } => Err(ObservationError::Identity(
            "integral partitions do not own selected entities",
        )),
    }
}
