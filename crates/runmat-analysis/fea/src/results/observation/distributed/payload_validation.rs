//! Checks that a partition contribution's payload matches what its partition target owns.
use super::*;

pub(super) fn validate_contribution_payload(
    partition: &ObservationPartition,
    contribution: &ObservationPartitionContribution,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    let storage_components = observation_result_storage_component_count(
        shape_component_count(&observation.result_value_shape)?,
        observation.spec.complex_representation,
    )?;
    match (&partition.target, &contribution.payload) {
        (
            ObservationPartitionTarget::Nodes { identities, .. },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(ObservationEntity::Node)
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "nodal observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Nodes { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant | SpatialReduction::Moment { .. })
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::ConstraintGroups { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant)
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::Elements {
                domain, identities, ..
            },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(|identity| {
                    super::super::contract::observation_element_entity(*domain, identity)
                })
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "element observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Points {
                sample_ordinals,
                identities,
            },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let super::super::ResolvedObservationTarget::Points { samples } = &observation.target
            else {
                return Err(ObservationError::Identity(
                    "point contribution does not match its resolved target",
                ));
            };
            let expected = sample_ordinals
                .iter()
                .map(|ordinal| {
                    let sample = samples.get(*ordinal).ok_or(ObservationError::Identity(
                        "point contribution exceeds its resolved target",
                    ))?;
                    Ok(ObservationEntity::Point {
                        identity: sample.identity,
                        ordinal: sample.ordinal,
                        position_m: sample.position_m,
                        source_element: sample.source_element_identity,
                    })
                })
                .collect::<Result<Vec<_>, ObservationError>>()?;
            if identities
                != &expected
                    .iter()
                    .map(|entity| match entity {
                        ObservationEntity::Point { identity, .. } => *identity,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>()
                || entities != &expected
                || values.len()
                    != sample_ordinals
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "point observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Connectors { identities, .. },
            ObservationPartitionPayload::SelectedValues { entities, values },
        ) => {
            let expected = identities
                .iter()
                .copied()
                .map(ObservationEntity::Connector)
                .collect::<Vec<_>>();
            if entities != &expected
                || values.len()
                    != identities
                        .len()
                        .checked_mul(storage_components)
                        .ok_or(ObservationError::Limit("partition result value count"))?
            {
                return Err(ObservationError::Identity(
                    "connector observation contribution differs from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::Connectors { indices, .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                measure,
            },
        ) if matches!(
            reduction_operation(observation),
            Ok(SpatialReduction::Resultant)
        ) && linear_sum.len() == storage_components
            && square_sum.len() == storage_components
            && *measure == indices.len() as f64 => {}
        (
            ObservationPartitionTarget::Nodes { .. }
            | ObservationPartitionTarget::Elements { .. }
            | ObservationPartitionTarget::Points { .. },
            ObservationPartitionPayload::ExtremaCandidates { entities, values },
        ) if spatial_extrema(observation).is_some() => {
            let owned = partition_selected_entities(partition, observation)?;
            let mut previous = None;
            for entity in entities {
                let Some(index) = owned.iter().position(|candidate| candidate == entity) else {
                    return Err(ObservationError::Identity(
                        "extrema candidate is not owned by its observation partition",
                    ));
                };
                if previous.is_some_and(|previous| index <= previous) {
                    return Err(ObservationError::Identity(
                        "extrema candidates are not in canonical partition order",
                    ));
                }
                previous = Some(index);
            }
            if entities.is_empty()
                || entities.len() != values.len()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(ObservationError::Identity(
                    "extrema candidate values differ from partition ownership",
                ));
            }
        }
        (
            ObservationPartitionTarget::BoundaryPair {
                face_indices,
                first_count,
                ..
            },
            ObservationPartitionPayload::PairedWeightedReduction {
                first_measure,
                second_measure,
                ..
            },
        ) if (*first_count > 0) == (*first_measure > 0.0)
            && (*first_count < face_indices.len()) == (*second_measure > 0.0) => {}
        (
            ObservationPartitionTarget::Volume { .. } | ObservationPartitionTarget::Boundary { .. },
            ObservationPartitionPayload::WeightedReduction {
                linear_sum,
                square_sum,
                ..
            },
        ) if linear_sum.len() == storage_components && square_sum.len() == storage_components => {}
        _ => {
            return Err(ObservationError::Identity(
                "observation contribution payload differs from partition target",
            ))
        }
    }
    Ok(())
}
