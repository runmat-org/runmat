//! Resolved observation target, numeric and storage contract checks.
use super::*;
pub(super) fn validate_resolved_observation(
    topology_digest: Digest,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    validate_output_numeric_contract(&observation.spec)?;
    if observation.sample_bounds.maximum_produced_samples == 0
        || observation.sample_bounds.maximum_retained_samples == 0
        || observation.sample_bounds.maximum_retained_samples
            > observation.sample_bounds.maximum_produced_samples
    {
        return Err(ObservationError::Invalid(
            "resolved observation sample bounds are invalid".into(),
        ));
    }
    let expected_sample_bounds = if observation.spec.sampling == SamplingPolicy::Final {
        ObservationSampleBounds {
            maximum_produced_samples: 1,
            maximum_retained_samples: 1,
        }
    } else {
        super::super::history::resolve_history_sample_bounds(
            &observation.spec.sampling,
            &observation.spec.storage.retention,
            observation.sample_bounds.maximum_produced_samples,
        )?
    };
    if observation.sample_bounds != expected_sample_bounds {
        return Err(ObservationError::Identity(
            "resolved observation sample bounds do not converge",
        ));
    }
    let source_components = shape_component_count(&observation.spec.value_shape)?;
    let spatial_extrema = observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| {
            matches!(
                reduction.spatial,
                SpatialReduction::Minimum | SpatialReduction::Maximum
            )
        });
    let has_spatial_reduction = observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    let has_discrete_reaction_reduction =
        observation
            .spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| {
                matches!(
                    reduction.spatial,
                    SpatialReduction::Resultant | SpatialReduction::Moment { .. }
                ) && matches!(
                    observation.spec.quantity,
                    runmat_analysis_core::ObservationQuantity::ReactionForce
                        | runmat_analysis_core::ObservationQuantity::ReactionMoment
                )
            });
    let component_projection_is_valid = match observation.component {
        ResolvedComponentProjection::All => true,
        ResolvedComponentProjection::Cartesian { axis } => axis < source_components,
        ResolvedComponentProjection::Magnitude3 => source_components == 3,
        ResolvedComponentProjection::VonMisesSymmetricTensor
        | ResolvedComponentProjection::MaximumPrincipalSymmetricTensor
        | ResolvedComponentProjection::MinimumPrincipalSymmetricTensor => source_components == 6,
    };
    if !component_projection_is_valid
        || observation.result_value_shape != observation.spec.result_value_shape()
        || shape_component_count(&observation.result_value_shape)? == 0
    {
        return Err(ObservationError::Invalid(
            "resolved result value shape is inconsistent".into(),
        ));
    }
    match &observation.target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => {
            if materialized_field_location(&observation.spec) != FieldLocation::Node
                || observation.source_entity_domain != RecoveredFieldEntityDomain::Nodes
                || (has_spatial_reduction && !spatial_extrema && !has_discrete_reaction_reduction)
                || indices.len() != identities.len()
                || indices.is_empty()
                || !strictly_increasing(indices)
                || !all_unique(identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved node target is empty, unordered, or semantically inconsistent".into(),
                ));
            }
        }
        ResolvedObservationTarget::Elements {
            domain,
            indices,
            identities,
        } => {
            if !matches!(
                materialized_field_location(&observation.spec),
                FieldLocation::Element
                    | FieldLocation::Cell
                    | FieldLocation::Face
                    | FieldLocation::IntegrationPoint
                    | FieldLocation::Edge
            ) || (materialized_field_location(&observation.spec)
                == FieldLocation::IntegrationPoint)
                != (*domain == RecoveredFieldEntityDomain::VolumeIntegrationPoints)
                || (materialized_field_location(&observation.spec) == FieldLocation::Edge)
                    != (*domain == RecoveredFieldEntityDomain::OrientedEdges)
                || (materialized_field_location(&observation.spec) == FieldLocation::Face)
                    != (*domain == RecoveredFieldEntityDomain::OrientedFaces)
                || observation.source_entity_domain != *domain
                || *domain == RecoveredFieldEntityDomain::Nodes
                || (has_spatial_reduction && !spatial_extrema)
                || indices.len() != identities.len()
                || indices.is_empty()
                || !strictly_increasing(indices)
                || !all_unique(identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved element target is empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
        ResolvedObservationTarget::Points { samples } => {
            let valid_ordinals = samples
                .iter()
                .enumerate()
                .all(|(ordinal, sample)| sample.ordinal == ordinal as u64);
            let valid_samples = samples.iter().all(|sample| {
                sample.identity != StableDigest::ZERO
                    && !sample.source_node_indices.is_empty()
                    && sample.source_node_indices.len() == sample.source_node_identities.len()
                    && sample.source_node_indices.len() == sample.shape_values.len()
                    && sample.position_m.iter().all(|value| value.is_finite())
                    && sample.shape_values.iter().all(|value| value.is_finite())
                    && (sample.shape_values.iter().sum::<f64>() - 1.0).abs() <= 1.0e-9
                    && derive_resolved_point_identity(topology_digest, sample)
                        .is_ok_and(|identity| identity == sample.identity)
            });
            if materialized_field_location(&observation.spec) != FieldLocation::Node
                || observation.spec.output_location != FieldLocation::Point
                || observation.source_entity_domain != RecoveredFieldEntityDomain::Nodes
                || observation.spec.recovery
                    != runmat_analysis_core::RecoveryPolicy::ShapeFunctionInterpolation
                || (has_spatial_reduction && !spatial_extrema)
                || samples.is_empty()
                || !valid_ordinals
                || !valid_samples
            {
                return Err(ObservationError::Invalid(
                    "resolved point samples are empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
        ResolvedObservationTarget::ConstraintGroups {
            indices,
            identities,
        } => {
            if observation.spec.source_location != FieldLocation::Constraint
                || observation.spec.output_location != FieldLocation::Global
                || observation.source_entity_domain != RecoveredFieldEntityDomain::ConstraintGroups
                || !has_discrete_reaction_reduction
                || indices.len() != 1
                || identities.len() != 1
                || identities[0] == StableDigest::ZERO
            {
                return Err(ObservationError::Invalid(
                    "resolved constraint target is not one canonical reaction-result group".into(),
                ));
            }
        }
        ResolvedObservationTarget::Connectors {
            indices,
            identities,
        } => {
            let reduction_is_valid = observation.spec.reduction.as_ref().is_none_or(|reduction| {
                matches!(
                    (observation.spec.quantity, reduction.spatial),
                    (
                        runmat_analysis_core::ObservationQuantity::ReactionForce,
                        SpatialReduction::Resultant
                    )
                )
            });
            if observation.spec.source_location != FieldLocation::Connector
                || observation.spec.output_location != FieldLocation::Connector
                || observation.source_entity_domain != RecoveredFieldEntityDomain::Connectors
                || !reduction_is_valid
                || indices.len() != 1
                || identities.len() != 1
                || identities[0] == StableDigest::ZERO
            {
                return Err(ObservationError::Invalid(
                    "resolved connector target is not one canonical connector".into(),
                ));
            }
        }
        ResolvedObservationTarget::Volume {
            element_indices,
            element_identities,
        } => {
            if observation.spec.reduction.is_none()
                || !matches!(
                    observation.source_entity_domain,
                    RecoveredFieldEntityDomain::Nodes | RecoveredFieldEntityDomain::VolumeElements
                )
                || element_indices.is_empty()
                || !strictly_increasing(element_indices)
                || element_indices.len() != element_identities.len()
                || !all_unique(element_identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved volume target is empty, unordered, or lacks a reduction".into(),
                ));
            }
        }
        ResolvedObservationTarget::Boundary {
            face_indices,
            face_identities,
        } => {
            // Nodal fields reduce over boundary faces at the face nodes; volume-element fields
            // are read in each face's owner element (their surface trace).
            let nodal = materialized_field_location(&observation.spec) == FieldLocation::Node
                && observation.source_entity_domain == RecoveredFieldEntityDomain::Nodes;
            let element_trace = materialized_field_location(&observation.spec)
                == FieldLocation::Element
                && observation.source_entity_domain == RecoveredFieldEntityDomain::VolumeElements;
            if !(nodal || element_trace)
                || observation.spec.reduction.is_none()
                || face_indices.is_empty()
                || !strictly_increasing(face_indices)
                || face_indices.len() != face_identities.len()
                || !all_unique(face_identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved boundary target is empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
        ResolvedObservationTarget::BoundaryPair {
            face_indices,
            face_identities,
            first_count,
        } => {
            let readable = matches!(
                (
                    materialized_field_location(&observation.spec),
                    observation.source_entity_domain
                ),
                (FieldLocation::Node, RecoveredFieldEntityDomain::Nodes)
                    | (
                        FieldLocation::Element | FieldLocation::Cell,
                        RecoveredFieldEntityDomain::VolumeElements
                    )
                    | (FieldLocation::Face, RecoveredFieldEntityDomain::OrientedFaces)
            );
            let (first, second) = face_indices.split_at((*first_count).min(face_indices.len()));
            if !readable
                || !observation.spec.spatial_reduction().is_paired()
                || first.is_empty()
                || second.is_empty()
                || !strictly_increasing(first)
                || !strictly_increasing(second)
                || first.iter().any(|index| second.binary_search(index).is_ok())
                || face_indices.len() != face_identities.len()
                || !all_unique(face_identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved paired boundary target is empty, unordered, overlapping, or semantically inconsistent"
                        .into(),
                ));
            }
        }
    }
    if observation.resolved_cost.operation_count == 0
        || observation.resolved_cost.peak_resident_bytes == 0
        || observation.resolved_cost.stored_bytes == 0
        || observation.resolved_cost.transfer_bytes == 0
        || observation.resolved_cost.collective_count != 0
    {
        return Err(ObservationError::Invalid(
            "resolved observation cost is incomplete or unsupported".into(),
        ));
    }
    Ok(())
}

pub(in crate::results::observation) fn materialized_field_location(
    spec: &ObservationSpec,
) -> FieldLocation {
    if spec.source_location == FieldLocation::Element
        && spec.output_location == FieldLocation::Node
        && spec.recovery == runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage
    {
        FieldLocation::Node
    } else {
        spec.source_location
    }
}

pub(in crate::results::observation) fn validate_output_numeric_contract(
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    let precision_matches = match spec.complex_representation {
        ComplexRepresentation::Complex => spec.storage.precision == SignalPrecision::Complex128,
        _ => spec.storage.precision == SignalPrecision::Real64,
    };
    if !precision_matches {
        return Err(ObservationError::Invalid(
            "observation precision does not match its requested real or complex representation"
                .into(),
        ));
    }
    if spec.complex_representation == ComplexRepresentation::SoundPressureLevel
        && (spec.quantity != runmat_analysis_core::ObservationQuantity::AcousticPressure
            || spec.value_shape != runmat_analysis_core::ObservationValueShape::Scalar
            || spec.reduction.as_ref().is_some_and(|reduction| {
                !matches!(
                    reduction.spatial,
                    SpatialReduction::Minimum | SpatialReduction::Maximum
                )
            }))
    {
        return Err(ObservationError::Unsupported(
            "sound pressure level applies to scalar acoustic pressure, unreduced or as a spatial extremum",
        ));
    }
    if spec.complex_representation == ComplexRepresentation::PhaseRadians
        && spec.reduction.is_some()
    {
        return Err(ObservationError::Unsupported(
            "phase requires a circular reduction rather than a linear spatial reduction",
        ));
    }
    if spec.complex_representation == ComplexRepresentation::Complex
        && spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| reduction.spatial == SpatialReduction::RootMeanSquare)
    {
        return Err(ObservationError::Unsupported(
            "complex root-mean-square reduction requires an explicit magnitude representation",
        ));
    }
    if spec.component == ObservationComponent::Magnitude
        && matches!(
            spec.complex_representation,
            ComplexRepresentation::PhaseRadians | ComplexRepresentation::Complex
        )
    {
        return Err(ObservationError::Unsupported(
            "vector magnitude requires a real, imaginary, or complex-magnitude representation",
        ));
    }
    Ok(())
}

pub(super) fn strictly_increasing(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub(super) fn all_unique(values: &[StableDigest]) -> bool {
    values
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        == values.len()
}
