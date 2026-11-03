use crate::algebra::CompensatedSum;
use crate::results::field::RecoveredFieldSample;
use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, ExtremaTiePolicy, SeriesReduction, SeriesWeighting,
    SpatialReduction,
};

use super::super::{
    ObservationEntity, ObservationError, ObservationExecutionPolicy, ResolvedObservationPlan,
};
use super::series_contract::{
    SeriesExtremumLocation, SeriesReductionContribution, SeriesReductionEvidence,
    SeriesReductionPayload, SeriesReductionResult, SeriesReductionResultIdentity,
    SeriesReductionSource,
};
use super::{ObservationHistory, ObservationHistorySample};

pub fn reduce_observation_history(
    plan: &ResolvedObservationPlan,
    history: &ObservationHistory,
    policy: ObservationExecutionPolicy,
) -> Result<SeriesReductionResult, ObservationError> {
    history.validate()?;
    let observation = plan
        .observations
        .iter()
        .find(|candidate| candidate.spec.observation_id == history.manifest().observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid("history observation is absent from its plan".into())
        })?;
    let reduction = observation.spec.reduction.as_ref().ok_or_else(|| {
        ObservationError::Invalid("history does not request a series reduction".into())
    })?;
    if reduction.series == SeriesReduction::None {
        return Err(ObservationError::Invalid(
            "history does not request a series reduction".into(),
        ));
    }
    if history.manifest().observed_sample_count as usize != history.samples().len() {
        return Err(ObservationError::Invalid(
            "series reduction requires the complete retained history".into(),
        ));
    }
    validate_series_inputs(history, reduction.series, reduction.spatial)?;

    let partition_count = history
        .samples()
        .len()
        .min(policy.maximum_partitions_per_observation().get() as usize);
    let contributions = canonical_sample_ranges(history.samples().len(), partition_count)
        .into_iter()
        .enumerate()
        .map(|(ordinal, range)| {
            SeriesReductionContribution::new(
                u32::try_from(ordinal)
                    .map_err(|_| ObservationError::Limit("series partition ordinal"))?,
                &history.samples()[range],
            )
        })
        .collect::<Result<Vec<_>, ObservationError>>()?;
    let evidence = SeriesReductionEvidence {
        source_history: history.manifest().identity,
        requested_partition_limit: policy.maximum_partitions_per_observation(),
        contributions,
    };
    validate_evidence_against_history(&evidence, history)?;
    let payload = reduce_values(history, reduction.series, reduction.extrema_ties)?;
    let mut result = SeriesReductionResult {
        schema_version: 1,
        identity: SeriesReductionResultIdentity::placeholder(),
        source_history: history.manifest().identity,
        observation_id: history.manifest().observation_id.clone(),
        operation: reduction.series,
        extrema_ties: reduction.extrema_ties,
        payload,
        evidence,
    };
    result.identity = result.derive_identity()?;
    result.validate_for(plan, history)?;
    Ok(result)
}

fn canonical_sample_ranges(count: usize, partitions: usize) -> Vec<std::ops::Range<usize>> {
    let base = count / partitions;
    let remainder = count % partitions;
    let mut start = 0;
    (0..partitions)
        .map(|ordinal| {
            let width = base + usize::from(ordinal < remainder);
            let range = start..start + width;
            start += width;
            range
        })
        .collect()
}

fn validate_series_inputs(
    history: &ObservationHistory,
    operation: SeriesReduction,
    spatial: SpatialReduction,
) -> Result<(), ObservationError> {
    // Spatial extrema have stable value lanes, but their entities are sample-local
    // governing locations. A different node may govern at each time/frequency.
    // Unreduced fields retain entity-indexed lanes and must keep their exact layout.
    let sample_local_entities = matches!(
        spatial,
        SpatialReduction::Minimum | SpatialReduction::Maximum
    );
    let first = &history.samples()[0].result;
    let first_values = first
        .field
        .as_host_f64()
        .ok_or(ObservationError::Unsupported(
            "series reduction requires host-resident f64 values",
        ))?;
    if first_values.is_empty() {
        return Err(ObservationError::Invalid(
            "series reduction requires nonempty samples".into(),
        ));
    }
    for sample in history.samples() {
        let values = sample
            .result
            .field
            .as_host_f64()
            .ok_or(ObservationError::Unsupported(
                "series reduction requires host-resident f64 values",
            ))?;
        if sample.result.field.shape != first.field.shape
            || (!sample_local_entities && sample.result.entities != first.entities)
            || values.len() != first_values.len()
        {
            return Err(ObservationError::Identity(
                "series samples do not share one canonical value layout",
            ));
        }
    }
    if history.manifest().complex_representation == ComplexRepresentation::Complex
        && !matches!(operation, SeriesReduction::Mean { .. })
    {
        return Err(ObservationError::Unsupported(
            "complex series require an explicit real view for this reduction",
        ));
    }
    if matches!(
        operation,
        SeriesReduction::RootMeanSquare { .. }
            | SeriesReduction::Mean { .. }
            | SeriesReduction::Variance { .. }
    ) && history.samples().len() < 2
    {
        return Err(ObservationError::Invalid(
            "statistical series reduction requires at least two samples".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_evidence_against_history(
    evidence: &SeriesReductionEvidence,
    history: &ObservationHistory,
) -> Result<(), ObservationError> {
    evidence.validate()?;
    if evidence.source_history != history.manifest().identity
        || evidence.contributions.is_empty()
        || evidence.contributions.len() > evidence.requested_partition_limit.get() as usize
    {
        return Err(ObservationError::Identity(
            "series-reduction evidence differs from its source history",
        ));
    }
    let lanes = history.samples()[0].result.field.element_count();
    let mut source_offset = 0usize;
    for (ordinal, contribution) in evidence.contributions.iter().enumerate() {
        contribution.validate()?;
        if contribution.ordinal as usize != ordinal
            || contribution.values.len() != contribution.sources.len().saturating_mul(lanes)
        {
            return Err(ObservationError::Identity(
                "series-reduction contribution shape differs from its source history",
            ));
        }
        for (local, source) in contribution.sources.iter().enumerate() {
            let sample = history
                .samples()
                .get(source_offset)
                .ok_or(ObservationError::Identity(
                    "series-reduction evidence has extra source samples",
                ))?;
            let values = sample.result.field.as_host_f64().unwrap();
            let start = local * lanes;
            if source.sample != sample.identity
                || source.coordinate != sample.coordinate
                || contribution.values[start..start + lanes] != *values
            {
                return Err(ObservationError::Identity(
                    "series-reduction contribution differs from its source sample",
                ));
            }
            source_offset += 1;
        }
    }
    if source_offset != history.samples().len() {
        return Err(ObservationError::Identity(
            "series-reduction evidence omits source samples",
        ));
    }
    Ok(())
}

pub(super) fn reduce_values(
    history: &ObservationHistory,
    operation: SeriesReduction,
    ties: ExtremaTiePolicy,
) -> Result<SeriesReductionPayload, ObservationError> {
    let lanes = history.samples()[0].result.field.element_count();
    let shape = history.samples()[0].result.field.shape.clone();
    let field_id = format!("{}.series", history.manifest().observation_id.0);
    match operation {
        SeriesReduction::Minimum | SeriesReduction::Maximum | SeriesReduction::AbsoluteMaximum => {
            let (values, extrema) = extrema_by_lane(history, operation, ties)?;
            Ok(SeriesReductionPayload::Values {
                field: AnalysisField::host_f64(field_id, history.samples()[0].result.field.layout.clone(), shape, values),
                extrema,
            })
        }
        SeriesReduction::Envelope => {
            let (minimum, minima) = extrema_by_lane(history, SeriesReduction::Minimum, ties)?;
            let (maximum, maxima) = extrema_by_lane(history, SeriesReduction::Maximum, ties)?;
            Ok(SeriesReductionPayload::Envelope {
                minimum: AnalysisField::host_f64(
                    format!("{field_id}.minimum"), history.samples()[0].result.field.layout.clone(),
                    shape.clone(),
                    minimum,
                ),
                maximum: AnalysisField::host_f64(format!("{field_id}.maximum"), history.samples()[0].result.field.layout.clone(), shape, maximum),
                minima,
                maxima,
            })
        }
        SeriesReduction::RootMeanSquare { weighting }
        | SeriesReduction::Mean { weighting }
        | SeriesReduction::Variance { weighting } => {
            let weights = series_weights(history, weighting)?;
            let mut linear = vec![CompensatedSum::default(); lanes];
            let mut total_weight = CompensatedSum::default();
            for (sample, weight) in history.samples().iter().zip(weights.iter().copied()) {
                total_weight.add(weight);
                for (lane, value) in sample
                    .result
                    .field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .copied()
                    .enumerate()
                {
                    linear[lane].add(weight * value);
                }
            }
            let total_weight = total_weight.total();
            if !total_weight.is_finite() || total_weight <= 0.0 {
                return Err(ObservationError::Invalid(
                    "series weights have no positive finite measure".into(),
                ));
            }
            let means = linear
                .into_iter()
                .map(|sum| sum.total() / total_weight)
                .collect::<Vec<_>>();
            let values = match operation {
                SeriesReduction::Mean { .. } => means,
                SeriesReduction::RootMeanSquare { .. } => {
                    weighted_squares(history, &weights, &vec![0.0; lanes])
                        .into_iter()
                        .map(|sum| (sum.total() / total_weight).sqrt())
                        .collect()
                }
                SeriesReduction::Variance { .. } => weighted_squares(history, &weights, &means)
                    .into_iter()
                    .map(|sum| sum.total() / total_weight)
                    .collect(),
                _ => unreachable!(),
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(ObservationError::Invalid(
                    "series reduction produced a non-finite statistic".into(),
                ));
            }
            Ok(SeriesReductionPayload::Values {
                field: AnalysisField::host_f64(field_id, history.samples()[0].result.field.layout.clone(), shape, values),
                extrema: Vec::new(),
            })
        }
        SeriesReduction::None => Err(ObservationError::Invalid(
            "series reduction cannot evaluate a none policy".into(),
        )),
    }
}

fn weighted_squares(
    history: &ObservationHistory,
    weights: &[f64],
    centers: &[f64],
) -> Vec<CompensatedSum> {
    let mut squares = vec![CompensatedSum::default(); centers.len()];
    for (sample, weight) in history.samples().iter().zip(weights.iter().copied()) {
        for ((sum, value), center) in squares
            .iter_mut()
            .zip(sample.result.field.as_host_f64().unwrap())
            .zip(centers)
        {
            let delta = value - center;
            sum.add(weight * delta * delta);
        }
    }
    squares
}

fn extrema_by_lane(
    history: &ObservationHistory,
    operation: SeriesReduction,
    ties: ExtremaTiePolicy,
) -> Result<(Vec<f64>, Vec<SeriesExtremumLocation>), ObservationError> {
    let lanes = history.samples()[0].result.field.element_count();
    let mut output = Vec::with_capacity(lanes);
    let mut locations = Vec::new();
    for lane in 0..lanes {
        let metric = |value: f64| match operation {
            SeriesReduction::AbsoluteMaximum => value.abs(),
            _ => value,
        };
        let extreme = history
            .samples()
            .iter()
            .map(|sample| metric(sample.result.field.as_host_f64().unwrap()[lane]))
            .reduce(|left, right| match operation {
                SeriesReduction::Minimum => left.min(right),
                SeriesReduction::Maximum | SeriesReduction::AbsoluteMaximum => left.max(right),
                _ => unreachable!(),
            })
            .unwrap();
        output.push(extreme);
        for sample in history.samples() {
            let value = sample.result.field.as_host_f64().unwrap()[lane];
            if metric(value) == extreme {
                locations.push(SeriesExtremumLocation {
                    lane_index: lane,
                    value,
                    source: SeriesReductionSource {
                        sample: sample.identity,
                        coordinate: sample.coordinate,
                    },
                    entity: entity_for_lane(sample, lane),
                });
                if ties == ExtremaTiePolicy::FirstCanonical {
                    break;
                }
            }
        }
    }
    Ok((output, locations))
}

fn entity_for_lane(sample: &ObservationHistorySample, lane: usize) -> Option<ObservationEntity> {
    let entity_count = sample.result.entities.len();
    if entity_count == 0 {
        return None;
    }
    let lanes_per_entity = sample.result.field.element_count() / entity_count;
    sample.result.entities.get(lane / lanes_per_entity).cloned()
}

fn series_weights(
    history: &ObservationHistory,
    weighting: SeriesWeighting,
) -> Result<Vec<f64>, ObservationError> {
    match weighting {
        SeriesWeighting::UniformSamples => Ok(vec![1.0; history.samples().len()]),
        SeriesWeighting::TrapezoidalIndependentDomain => {
            let coordinates = history
                .samples()
                .iter()
                .map(|sample| continuous_coordinate(sample.coordinate))
                .collect::<Result<Vec<_>, ObservationError>>()?;
            let mut weights = vec![0.0; coordinates.len()];
            for (index, pair) in coordinates.windows(2).enumerate() {
                let interval = pair[1] - pair[0];
                if !interval.is_finite() || interval <= 0.0 {
                    return Err(ObservationError::Invalid(
                        "trapezoidal series coordinates must be strictly increasing".into(),
                    ));
                }
                weights[index] += 0.5 * interval;
                weights[index + 1] += 0.5 * interval;
            }
            Ok(weights)
        }
    }
}

fn continuous_coordinate(sample: RecoveredFieldSample) -> Result<f64, ObservationError> {
    match sample {
        RecoveredFieldSample::TimeSeconds(value)
        | RecoveredFieldSample::FrequencyHertz(value)
        | RecoveredFieldSample::LoadFactor(value)
        | RecoveredFieldSample::StudyParameterValue(value) => Ok(value),
        RecoveredFieldSample::Final | RecoveredFieldSample::ModeIndex(_) => {
            Err(ObservationError::Invalid(
                "trapezoidal weighting requires a continuous independent domain".into(),
            ))
        }
    }
}
