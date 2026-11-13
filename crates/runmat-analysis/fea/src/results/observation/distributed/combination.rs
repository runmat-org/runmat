//! Canonical combination of already validated original observation partitions.
use super::*;
use crate::algebra::CompensatedSum;

pub(crate) fn combine_observation_partitions(
    plan: &ResolvedObservationPlan,
    partition_plan: &ObservationPartitionPlan,
    mut contributions: Vec<ObservationPartitionContribution>,
) -> Result<ObservationResult, ObservationError> {
    partition_plan.validate_for(plan)?;
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == partition_plan.observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("partitioned observation is absent from its plan".into())
        })?;
    contributions.sort_by_key(|contribution| contribution.partition_ordinal);
    if contributions.len() != partition_plan.partitions.len() {
        return Err(ObservationError::Identity(
            "observation reduction is missing partition contributions",
        ));
    }
    for (partition, contribution) in partition_plan.partitions.iter().zip(&contributions) {
        contribution.validate()?;
        if contribution.partition_plan != partition_plan.identity
            || contribution.partition != partition.identity
            || contribution.partition_ordinal != partition.ordinal
            || contribution.observation_id != partition_plan.observation_id
            || contribution.source_field_definition != observation.source_field_definition_identity
        {
            return Err(ObservationError::Identity(
                "observation contribution differs from its partition plan",
            ));
        }
        validate_contribution_payload(partition, contribution, observation)?;
    }
    let first_source = contributions[0].source_field;
    let first_sample = contributions[0].sample;
    let first_event_attribution = contributions[0].event_attribution;
    if contributions.iter().any(|contribution| {
        contribution.source_field != first_source
            || contribution.sample != first_sample
            || contribution.event_attribution != first_event_attribution
    }) {
        return Err(ObservationError::Identity(
            "observation contributions name different source fields, samples, or events",
        ));
    }
    let source_partitions = contributions
        .iter()
        .flat_map(|contribution| contribution.source_partitions.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let contribution_identities = contributions
        .iter()
        .map(|contribution| contribution.identity)
        .collect::<Vec<_>>();
    let accelerations = contributions
        .iter()
        .filter_map(|contribution| contribution.acceleration.clone())
        .collect::<Vec<_>>();
    let combination = match &contributions[0].payload {
        ObservationPartitionPayload::SelectedValues { .. } => {
            ObservationPartitionCombination::CanonicalConcatenation
        }
        ObservationPartitionPayload::ExtremaCandidates { .. } => {
            ObservationPartitionCombination::CanonicalExtremaReduction {
                collective_count: u32::try_from(contributions.len().saturating_sub(1))
                    .map_err(|_| ObservationError::Limit("observation collective count"))?,
            }
        }
        ObservationPartitionPayload::WeightedReduction { .. }
        | ObservationPartitionPayload::PairedWeightedReduction { .. } => {
            ObservationPartitionCombination::CanonicalCompensatedReduction {
                collective_count: u32::try_from(contributions.len().saturating_sub(1))
                    .map_err(|_| ObservationError::Limit("observation collective count"))?,
            }
        }
    };
    let (values, entities) = match &contributions[0].payload {
        ObservationPartitionPayload::SelectedValues { .. } => {
            let mut entities = Vec::new();
            let mut values = Vec::new();
            for contribution in contributions {
                let ObservationPartitionPayload::SelectedValues {
                    entities: local_entities,
                    values: local_values,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                entities.extend(local_entities);
                values.extend(local_values);
            }
            (values, entities)
        }
        ObservationPartitionPayload::WeightedReduction { .. } => {
            let operation = reduction_operation(observation)?;
            let component_count = observation_result_storage_component_count(
                shape_component_count(&observation.result_value_shape)?,
                observation.spec.complex_representation,
            )?;
            let mut linear = vec![CompensatedSum::default(); component_count];
            let mut square = vec![CompensatedSum::default(); component_count];
            let mut measure = CompensatedSum::default();
            for contribution in contributions {
                let ObservationPartitionPayload::WeightedReduction {
                    linear_sum,
                    square_sum,
                    measure: local_measure,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                if linear_sum.len() != component_count || square_sum.len() != component_count {
                    return Err(ObservationError::Identity(
                        "observation partition component count differs from its plan",
                    ));
                }
                for (total, local) in linear.iter_mut().zip(linear_sum) {
                    total.add(local);
                }
                for (total, local) in square.iter_mut().zip(square_sum) {
                    total.add(local);
                }
                measure.add(local_measure);
            }
            let measure = measure.total();
            if !measure.is_finite() || measure <= 0.0 {
                return Err(ObservationError::Invalid(
                    "reduction target has no positive finite measure".into(),
                ));
            }
            let values = match operation {
                SpatialReduction::Integral | SpatialReduction::NormalFlux => {
                    linear.into_iter().map(|sum| sum.total()).collect()
                }
                SpatialReduction::Average => linear
                    .into_iter()
                    .map(|sum| sum.total() / measure)
                    .collect(),
                SpatialReduction::RootMeanSquare => square
                    .into_iter()
                    .map(|sum| (sum.total() / measure).sqrt())
                    .collect(),
                SpatialReduction::EuclideanNorm => {
                    let [sum] = square.as_slice() else {
                        return Err(ObservationError::Identity(
                            "Euclidean-norm contribution is not scalar",
                        ));
                    };
                    vec![sum.total().sqrt()]
                }
                SpatialReduction::Resultant | SpatialReduction::Moment { .. } => {
                    linear.into_iter().map(CompensatedSum::total).collect()
                }
                _ => {
                    return Err(ObservationError::Unsupported(
                        "unsupported partitioned spatial reduction",
                    ))
                }
            };
            (values, Vec::new())
        }
        ObservationPartitionPayload::PairedWeightedReduction { .. } => {
            let mut sums = [
                CompensatedSum::default(),
                CompensatedSum::default(),
                CompensatedSum::default(),
                CompensatedSum::default(),
            ];
            for contribution in contributions {
                let ObservationPartitionPayload::PairedWeightedReduction {
                    first_sum,
                    first_measure,
                    second_sum,
                    second_measure,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                for (total, local) in
                    sums.iter_mut()
                        .zip([first_sum, first_measure, second_sum, second_measure])
                {
                    total.add(local);
                }
            }
            let [first_sum, first_measure, second_sum, second_measure] =
                sums.map(CompensatedSum::total);
            if !(first_measure.is_finite()
                && first_measure > 0.0
                && second_measure.is_finite()
                && second_measure > 0.0)
            {
                return Err(ObservationError::Invalid(
                    "a paired reduction needs both regions to have a positive area".into(),
                ));
            }
            let value = match reduction_operation(observation)? {
                SpatialReduction::MeanDifference => {
                    first_sum / first_measure - second_sum / second_measure
                }
                // Incident port power over transmitted port power, both in watts.
                SpatialReduction::TransmissionLoss => {
                    if !(first_sum > 0.0 && second_sum > 0.0) {
                        return Err(ObservationError::Invalid(
                            "transmission loss needs a positive incident and transmitted port power; the incident region must carry an incident wave and the transmitted region an anechoic (radiation) boundary".into(),
                        ));
                    }
                    10.0 * (first_sum / second_sum).log10()
                }
                _ => {
                    return Err(ObservationError::Unsupported(
                        "paired contributions require a paired spatial reduction",
                    ))
                }
            };
            if !value.is_finite() {
                return Err(ObservationError::Invalid(
                    "paired reduction produced a non-finite value".into(),
                ));
            }
            (vec![value], Vec::new())
        }
        ObservationPartitionPayload::ExtremaCandidates { .. } => {
            let mut entities = Vec::new();
            let mut values = Vec::new();
            for contribution in contributions {
                let ObservationPartitionPayload::ExtremaCandidates {
                    entities: local_entities,
                    values: local_values,
                } = contribution.payload
                else {
                    return Err(ObservationError::Identity(
                        "observation partitions contain different payload kinds",
                    ));
                };
                entities.extend(local_entities);
                values.extend(local_values);
            }
            let (entities, values) = extrema_values(observation, entities, values)?;
            (values, entities)
        }
    };
    let value_shape = observation.result_value_shape.clone();
    let shape = observation_result_storage_shape(
        (!entities.is_empty()).then_some(entities.len()),
        &value_shape,
        observation.spec.complex_representation,
    )?;
    let partition_evidence =
        ObservationPartitionEvidence::new(ObservationPartitionEvidenceInput {
            partition_plan: partition_plan.identity,
            source_field: first_source,
            sample: first_sample,
            event_attribution: first_event_attribution,
            source_partitions,
            contributions: contribution_identities,
            accelerations,
            combination,
        })?;
    let mut result = ObservationResult {
        schema_version: 1,
        logical_identity: ObservationResultIdentity(Digest::sha256([])),
        plan_identity: plan.identity,
        topology_digest: plan.topology_digest,
        observation_id: observation.spec.observation_id.clone(),
        sample: first_sample,
        quantity: observation.spec.quantity,
        location: observation.spec.output_location,
        value_shape,
        complex_representation: observation.spec.complex_representation,
        entities,
        field: AnalysisField::host_f64(
            observation.spec.observation_id.0.clone(),
            runmat_analysis_core::AnalysisFieldLayout::new(
                observation.spec.quantity,
                observation.spec.output_location,
                observation.result_value_shape.clone(),
            )
            .representation(observation.spec.complex_representation),
            shape,
            values,
        ),
        partial_access: observation.spec.storage.partial_access,
        partition_evidence,
    };
    result.logical_identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}
