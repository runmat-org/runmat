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
            evaluate_observation_event_sample, evaluate_observation_sample,
            evaluate_structural_observation_event_sample, evaluate_structural_observation_sample,
            reduce_observation_history, resolve_observation_plan,
            resolve_structural_observation_plan,
            resolve_structural_observation_plan_with_connectors, sample_matches_policy,
            BoundedObservationHistory, ObservationError, ObservationEventAttribution,
            ObservationExecutionPolicy, ObservationHistory, ObservationLimits, ObservationTopology,
            SeriesReductionResult,
        },
    },
    topology::AnalysisProjection,
};

mod schedule;
pub(crate) use schedule::{
    history_schedule_construction_resident_bytes, required_bounded_time_history_samples,
    required_bounded_time_history_samples_admitted, required_load_factor_history_samples,
    required_mode_history_indices, required_time_history_samples,
};

pub(crate) struct HistoryFieldSeries<'a> {
    pub quantity: ObservationQuantity,
    pub source_field_id: &'a str,
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
        connector_plan: Option<&'a crate::connectors::CanonicalStructuralConnectorPlan>,
    },
}

#[derive(Default)]
pub(crate) struct EvaluatedObservationHistories {
    pub histories: Vec<ObservationHistory>,
    pub series_reductions: Vec<SeriesReductionResult>,
}

pub(crate) fn evaluate_field_histories(
    step: &ProcedureStep,
    topology: HistoryTopology<'_>,
    coordinates: &[RecoveredFieldSample],
    events: Option<&[ObservationEventAttribution]>,
    series: &[HistoryFieldSeries<'_>],
) -> Result<EvaluatedObservationHistories, ObservationError> {
    let mut history_step = step.clone();
    history_step
        .observations
        .retain(|request| matches!(request, ObservationRequest::History(_)));
    if history_step.observations.is_empty() {
        return Ok(EvaluatedObservationHistories {
            histories: Vec::new(),
            series_reductions: Vec::new(),
        });
    }
    if events.is_none()
        && history_step.observations.iter().any(|request| {
            matches!(
                request,
                ObservationRequest::History(runmat_analysis_core::ObservationSpec {
                    sampling: SamplingPolicy::Event(_),
                    ..
                })
            )
        })
    {
        return Err(ObservationError::Unsupported(
            "this solver pipeline does not own an event detector for the requested history",
        ));
    }
    let events = events.unwrap_or_default();
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
    for (index, attribution) in events.iter().copied().enumerate() {
        attribution.validate()?;
        if !coordinates.contains(&attribution.sample()) {
            return Err(ObservationError::Invalid(
                "event attribution names a state absent from the accepted solver coordinates"
                    .into(),
            ));
        }
        if events[..index].contains(&attribution) {
            return Err(ObservationError::Invalid(
                "event attribution inventory contains a duplicate event occurrence".into(),
            ));
        }
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
            let topology = ObservationTopology::from_projection(projection)?;
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
        HistoryTopology::Structural {
            model,
            mesh,
            connector_plan,
        } => {
            let topology = ObservationTopology::derive(model, mesh)?;
            let constraint_plan =
                crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
                    model, mesh,
                )
                .map_err(|error| ObservationError::Invalid(error.to_string()))?;
            let mut entity_counts = topology.entity_counts();
            entity_counts.constraint_groups = constraint_plan.groups.len();
            entity_counts.connectors = connector_plan.map_or(0, |plan| plan.entries.len());
            let plan = if let Some(connector_plan) = connector_plan {
                resolve_structural_observation_plan_with_connectors(
                    &history_step,
                    model,
                    mesh,
                    &constraint_plan,
                    connector_plan,
                    &definitions,
                    ObservationLimits::default(),
                )?
            } else {
                resolve_structural_observation_plan(
                    &history_step,
                    model,
                    mesh,
                    &constraint_plan,
                    &definitions,
                    ObservationLimits::default(),
                )?
            };
            (plan, topology.digest(), entity_counts)
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
            if let SamplingPolicy::Event(expected_event) = &observation.spec.sampling {
                for attribution in events.iter().copied().filter(|attribution| {
                    attribution.sample() == coordinate && attribution.event() == *expected_event
                }) {
                    let sample = match topology {
                        HistoryTopology::Solid(projection) => evaluate_observation_event_sample(
                            &plan,
                            observation_id,
                            projection,
                            &catalog,
                            attribution,
                        )?,
                        HistoryTopology::Structural { model, mesh, .. } => {
                            evaluate_structural_observation_event_sample(
                                &plan,
                                observation_id,
                                model,
                                mesh,
                                &catalog,
                                attribution,
                            )?
                        }
                    };
                    history.push_event(sample)?;
                }
            } else if sample_matches_policy(&observation.spec.sampling, coordinate)? {
                let result = match topology {
                    HistoryTopology::Solid(projection) => evaluate_observation_sample(
                        &plan,
                        observation_id,
                        projection,
                        &catalog,
                        coordinate,
                    )?,
                    HistoryTopology::Structural { model, mesh, .. } => {
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
    }

    let histories = histories
        .into_iter()
        .map(|(_, history)| history.finish())
        .collect::<Result<Vec<_>, _>>()?;
    let series_reductions = histories
        .iter()
        .filter(|history| {
            plan.observations
                .iter()
                .find(|observation| {
                    observation.spec.observation_id == history.manifest().observation_id
                })
                .and_then(|observation| observation.spec.reduction.as_ref())
                .is_some_and(|reduction| {
                    reduction.series != runmat_analysis_core::SeriesReduction::None
                })
        })
        .map(|history| {
            reduce_observation_history(&plan, history, ObservationExecutionPolicy::scalable())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EvaluatedObservationHistories {
        histories,
        series_reductions,
    })
}
