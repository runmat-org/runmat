use runmat_analysis_core::{
    AnalysisField, AnalysisModel, FieldLocation, ObservationQuantity, ObservationRequest,
    ObservationValueShape, ProcedureStep, SamplingPolicy,
};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    results::{
        field::{
            RecoveredField, RecoveredFieldCatalog, RecoveredFieldDefinition,
            RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
        },
        observation::{
            evaluate_observation_sample, evaluate_structural_observation_sample,
            exact_scheduled_coordinates, resolve_observation_plan,
            resolve_structural_observation_plan, sample_matches_policy, BoundedObservationHistory,
            ObservationError, ObservationHistory, ObservationLimits, StructuralObservationTopology,
        },
    },
    topology::AnalysisProjection,
};

pub(crate) fn required_time_history_samples(
    step: &ProcedureStep,
) -> Result<Vec<f64>, ObservationError> {
    required_scalar_history_samples(step, ScalarHistoryDomain::TimeSeconds)
}

pub(crate) fn required_load_factor_history_samples(
    step: &ProcedureStep,
) -> Result<Vec<f64>, ObservationError> {
    required_scalar_history_samples(step, ScalarHistoryDomain::LoadFactor)
}

pub(crate) fn required_mode_history_indices(
    step: &ProcedureStep,
) -> Result<Vec<usize>, ObservationError> {
    let coordinates = required_history_coordinates(step)?;
    coordinates
        .into_iter()
        .map(|coordinate| match coordinate {
            RecoveredFieldSample::ModeIndex(index) => Ok(index),
            _ => Err(ObservationError::Invalid(
                "modal history sampling differs from the eigenvalue domain".into(),
            )),
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ScalarHistoryDomain {
    TimeSeconds,
    LoadFactor,
}

fn required_scalar_history_samples(
    step: &ProcedureStep,
    domain: ScalarHistoryDomain,
) -> Result<Vec<f64>, ObservationError> {
    let coordinates = required_history_coordinates(step)?;
    let mut samples = coordinates
        .into_iter()
        .map(|coordinate| match (domain, coordinate) {
            (ScalarHistoryDomain::TimeSeconds, RecoveredFieldSample::TimeSeconds(value))
            | (ScalarHistoryDomain::LoadFactor, RecoveredFieldSample::LoadFactor(value)) => {
                Ok(value)
            }
            (ScalarHistoryDomain::TimeSeconds, _) => Err(ObservationError::Invalid(
                "transient history sampling differs from the time domain".into(),
            )),
            (ScalarHistoryDomain::LoadFactor, _) => Err(ObservationError::Invalid(
                "nonlinear history sampling differs from the load-factor domain".into(),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    samples.sort_by(f64::total_cmp);
    samples.dedup_by(|left, right| left.to_bits() == right.to_bits());
    Ok(samples)
}

fn required_history_coordinates(
    step: &ProcedureStep,
) -> Result<Vec<RecoveredFieldSample>, ObservationError> {
    let mut samples = Vec::new();
    for request in &step.observations {
        let ObservationRequest::History(spec) = request else {
            continue;
        };
        match &spec.sampling {
            SamplingPolicy::EveryAcceptedIncrement => {}
            SamplingPolicy::TimeSeconds(_)
            | SamplingPolicy::TimeIntervalSeconds { .. }
            | SamplingPolicy::FrequencyHertz(_)
            | SamplingPolicy::FrequencyIntervalHertz { .. }
            | SamplingPolicy::LoadFactors(_)
            | SamplingPolicy::LoadFactorInterval { .. }
            | SamplingPolicy::ModeIndices(_) => {
                let coordinates = exact_scheduled_coordinates(
                    &spec.sampling,
                    step.accuracy.resources.maximum_samples,
                )?
                .ok_or_else(|| {
                    ObservationError::Invalid(
                        "exact time history did not produce an exact schedule".into(),
                    )
                })?;
                samples.extend(coordinates);
            }
            SamplingPolicy::Adaptive { .. } => {
                return Err(ObservationError::Unsupported(
                    "adaptive history sampling requires response-aware sample selection",
                ));
            }
            SamplingPolicy::Event(_) => {
                return Err(ObservationError::Unsupported(
                    "event histories require solver-attributed event coordinates",
                ));
            }
            SamplingPolicy::Final => {
                return Err(ObservationError::Invalid(
                    "history requests require a sampled series rather than one final state".into(),
                ));
            }
        }
    }
    samples.sort_by(compare_coordinates);
    samples.dedup();
    Ok(samples)
}

fn compare_coordinates(
    left: &RecoveredFieldSample,
    right: &RecoveredFieldSample,
) -> std::cmp::Ordering {
    match (*left, *right) {
        (RecoveredFieldSample::TimeSeconds(left), RecoveredFieldSample::TimeSeconds(right))
        | (
            RecoveredFieldSample::FrequencyHertz(left),
            RecoveredFieldSample::FrequencyHertz(right),
        )
        | (RecoveredFieldSample::LoadFactor(left), RecoveredFieldSample::LoadFactor(right)) => {
            left.total_cmp(&right)
        }
        (RecoveredFieldSample::ModeIndex(left), RecoveredFieldSample::ModeIndex(right)) => {
            left.cmp(&right)
        }
        (left, right) => coordinate_rank(left).cmp(&coordinate_rank(right)),
    }
}

fn coordinate_rank(coordinate: RecoveredFieldSample) -> u8 {
    match coordinate {
        RecoveredFieldSample::Final => 0,
        RecoveredFieldSample::TimeSeconds(_) => 1,
        RecoveredFieldSample::FrequencyHertz(_) => 2,
        RecoveredFieldSample::LoadFactor(_) => 3,
        RecoveredFieldSample::ModeIndex(_) => 4,
    }
}

pub(crate) struct HistoryFieldSeries<'a> {
    pub quantity: ObservationQuantity,
    pub source_field_id: &'static str,
    pub location: FieldLocation,
    pub entity_domain: RecoveredFieldEntityDomain,
    pub value_shape: ObservationValueShape,
    pub numeric_representation: RecoveredFieldNumericRepresentation,
    pub snapshots: &'a [AnalysisField],
}

#[derive(Clone, Copy)]
pub(crate) enum HistoryTopology<'a> {
    Solid(&'a AnalysisProjection),
    Structural {
        model: &'a AnalysisModel,
        mesh: Option<&'a SolverMeshArtifact>,
    },
}

pub(crate) fn evaluate_field_histories(
    step: &ProcedureStep,
    topology: HistoryTopology<'_>,
    coordinates: &[RecoveredFieldSample],
    series: &[HistoryFieldSeries<'_>],
) -> Result<Vec<ObservationHistory>, ObservationError> {
    let mut history_step = step.clone();
    history_step
        .observations
        .retain(|request| matches!(request, ObservationRequest::History(_)));
    if history_step.observations.is_empty() {
        return Ok(Vec::new());
    }
    if coordinates.is_empty()
        || series.is_empty()
        || series
            .iter()
            .any(|source| source.snapshots.len() != coordinates.len())
    {
        return Err(ObservationError::Invalid(
            "history source series do not cover the accepted solver coordinates".into(),
        ));
    }

    let definitions = series
        .iter()
        .map(|source| {
            RecoveredFieldDefinition::new(
                source.source_field_id,
                source.quantity,
                source.location,
                source.entity_domain,
                source.value_shape.clone(),
                source.numeric_representation,
            )
            .map_err(|error| ObservationError::Invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (plan, topology_digest, entity_counts) = match topology {
        HistoryTopology::Solid(projection) => {
            let topology = StructuralObservationTopology::from_projection(projection);
            (
                resolve_observation_plan(
                    &history_step,
                    projection,
                    &definitions,
                    ObservationLimits::default(),
                )?,
                projection.projection_digest,
                topology.entity_counts(),
            )
        }
        HistoryTopology::Structural { model, mesh } => {
            let topology = StructuralObservationTopology::derive(model, mesh)?;
            (
                resolve_structural_observation_plan(
                    &history_step,
                    model,
                    mesh,
                    &definitions,
                    ObservationLimits::default(),
                )?,
                topology.digest(),
                topology.entity_counts(),
            )
        }
    };
    let mut histories = plan
        .observations
        .iter()
        .map(|observation| {
            BoundedObservationHistory::new(&plan, &observation.spec.observation_id)
                .map(|history| (observation.spec.observation_id.clone(), history))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for (sample_index, &coordinate) in coordinates.iter().enumerate() {
        let fields = definitions
            .iter()
            .zip(series)
            .map(|(definition, source)| {
                let mut storage = source.snapshots[sample_index].clone();
                storage.field_id = definition.field_id().to_owned();
                RecoveredField::new(definition.clone(), coordinate, storage)
                    .map_err(|error| ObservationError::Invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let catalog =
            RecoveredFieldCatalog::new_for_topology(topology_digest, entity_counts, fields)
                .map_err(|error| ObservationError::Invalid(error.to_string()))?;
        for (observation_id, history) in &mut histories {
            let observation = plan
                .observations
                .iter()
                .find(|observation| &observation.spec.observation_id == observation_id)
                .ok_or(ObservationError::Identity(
                    "history builder is absent from its resolved plan",
                ))?;
            if !sample_matches_policy(&observation.spec.sampling, coordinate)? {
                continue;
            }
            let result = match topology {
                HistoryTopology::Solid(projection) => evaluate_observation_sample(
                    &plan,
                    observation_id,
                    projection,
                    &catalog,
                    coordinate,
                )?,
                HistoryTopology::Structural { model, mesh } => {
                    evaluate_structural_observation_sample(
                        &plan,
                        observation_id,
                        model,
                        mesh,
                        &catalog,
                        coordinate,
                    )?
                }
            };
            history.push(coordinate, result)?;
        }
    }

    histories
        .into_iter()
        .map(|(_, history)| history.finish())
        .collect()
}
