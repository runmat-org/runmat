use runmat_analysis_core::{AnalysisModel, ObservationId, SamplingPolicy};
use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    results::field::{RecoveredFieldCatalog, RecoveredFieldSample},
    topology::AnalysisProjection,
};

use super::contract::{
    materialized_field_location, ObservationError, ObservationResult, ResolvedObservation,
    ResolvedObservationPlan, ResolvedObservationTarget,
};
use super::history::{
    sample_matches_domain, sample_matches_history_policy, ObservationEventAttribution,
    ObservationHistorySample,
};
use super::topology::ObservationTopology;

pub fn evaluate_observations(
    plan: &ResolvedObservationPlan,
    projection: &AnalysisProjection,
    fields: &RecoveredFieldCatalog,
) -> Result<Vec<ObservationResult>, ObservationError> {
    let topology = ObservationTopology::from_projection_for(
        projection,
        ObservationTopology::needs_faces(fields.fields().iter().map(|field| field.definition())),
    )?;
    evaluate_observations_for_topology(plan, &topology, fields)
}

pub fn evaluate_structural_observations(
    plan: &ResolvedObservationPlan,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    fields: &RecoveredFieldCatalog,
) -> Result<Vec<ObservationResult>, ObservationError> {
    let topology = ObservationTopology::derive(model, mesh)?;
    evaluate_observations_for_topology(plan, &topology, fields)
}

pub(super) fn evaluate_observations_for_topology(
    plan: &ResolvedObservationPlan,
    topology: &ObservationTopology,
    fields: &RecoveredFieldCatalog,
) -> Result<Vec<ObservationResult>, ObservationError> {
    if plan.topology_digest != topology.digest() {
        return Err(ObservationError::Identity(
            "observation plan names a different topology projection",
        ));
    }
    if fields.topology_digest() != topology.digest() {
        return Err(ObservationError::Identity(
            "recovered field catalog names a different topology projection",
        ));
    }
    validate_plan_targets(plan, topology)?;
    if plan
        .observations
        .iter()
        .any(|observation| observation.spec.sampling != SamplingPolicy::Final)
    {
        return Err(ObservationError::Unsupported(
            "sampled observations require bounded history evaluation",
        ));
    }
    plan.observations
        .iter()
        .map(|observation| {
            let physical = terminal_physical_sample(&plan.independent_domain).filter(|sample| {
                fields
                    .get(observation.source_field_definition_identity, *sample)
                    .is_some()
            });
            if physical.is_some()
                && fields
                    .get(
                        observation.source_field_definition_identity,
                        RecoveredFieldSample::Final,
                    )
                    .is_some()
            {
                return Err(ObservationError::Identity(
                    "final observation has ambiguous terminal field coordinates",
                ));
            }
            evaluate_sample(
                plan,
                observation,
                topology,
                fields,
                physical.unwrap_or(RecoveredFieldSample::Final),
                None,
            )
        })
        .collect()
}

pub fn evaluate_observation_sample(
    plan: &ResolvedObservationPlan,
    observation_id: &ObservationId,
    projection: &AnalysisProjection,
    fields: &RecoveredFieldCatalog,
    sample: RecoveredFieldSample,
) -> Result<ObservationResult, ObservationError> {
    let topology = ObservationTopology::from_projection_for(
        projection,
        ObservationTopology::needs_faces(fields.fields().iter().map(|field| field.definition())),
    )?;
    evaluate_observation_sample_for_topology(plan, observation_id, &topology, fields, sample, None)
}

pub fn evaluate_observation_event_sample(
    plan: &ResolvedObservationPlan,
    observation_id: &ObservationId,
    projection: &AnalysisProjection,
    fields: &RecoveredFieldCatalog,
    attribution: ObservationEventAttribution,
) -> Result<ObservationHistorySample, ObservationError> {
    let topology = ObservationTopology::from_projection_for(
        projection,
        ObservationTopology::needs_faces(fields.fields().iter().map(|field| field.definition())),
    )?;
    let sample = attribution.sample();
    let result = evaluate_observation_sample_for_topology(
        plan,
        observation_id,
        &topology,
        fields,
        sample,
        Some(attribution),
    )?;
    ObservationHistorySample::new_with_attribution(sample, Some(attribution), result)
}

pub fn evaluate_structural_observation_sample(
    plan: &ResolvedObservationPlan,
    observation_id: &ObservationId,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    fields: &RecoveredFieldCatalog,
    sample: RecoveredFieldSample,
) -> Result<ObservationResult, ObservationError> {
    let topology = ObservationTopology::derive(model, mesh)?;
    evaluate_observation_sample_for_topology(plan, observation_id, &topology, fields, sample, None)
}

pub fn evaluate_structural_observation_event_sample(
    plan: &ResolvedObservationPlan,
    observation_id: &ObservationId,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    fields: &RecoveredFieldCatalog,
    attribution: ObservationEventAttribution,
) -> Result<ObservationHistorySample, ObservationError> {
    let topology = ObservationTopology::derive(model, mesh)?;
    let sample = attribution.sample();
    let result = evaluate_observation_sample_for_topology(
        plan,
        observation_id,
        &topology,
        fields,
        sample,
        Some(attribution),
    )?;
    ObservationHistorySample::new_with_attribution(sample, Some(attribution), result)
}

fn evaluate_observation_sample_for_topology(
    plan: &ResolvedObservationPlan,
    observation_id: &ObservationId,
    topology: &ObservationTopology,
    fields: &RecoveredFieldCatalog,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationResult, ObservationError> {
    if plan.topology_digest != topology.digest() || fields.topology_digest() != topology.digest() {
        return Err(ObservationError::Identity(
            "sampled observation inputs name different topology projections",
        ));
    }
    validate_plan_targets(plan, topology)?;
    let observation = plan
        .observations
        .iter()
        .find(|observation| &observation.spec.observation_id == observation_id)
        .ok_or_else(|| {
            ObservationError::Invalid(
                "sampled observation identifier is absent from its resolved plan".into(),
            )
        })?;
    if observation.spec.sampling == SamplingPolicy::Final {
        return Err(ObservationError::Invalid(
            "a final-only observation cannot be appended to a history".into(),
        ));
    }
    if !resolved_observation_accepts_sample(plan, observation, sample, event_attribution)? {
        return Err(ObservationError::Invalid(
            "sample coordinate differs from the resolved observation schedule".into(),
        ));
    }
    evaluate_sample(
        plan,
        observation,
        topology,
        fields,
        sample,
        event_attribution,
    )
}

/// Returns whether one resolved observation admits the supplied physical sample.
///
/// This is the single schedule boundary shared by local evaluation and distributed fan-out. A
/// false result means no task should be created; malformed attribution or contracts remain errors.
pub fn resolved_observation_accepts_sample(
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<bool, ObservationError> {
    if observation.spec.sampling == SamplingPolicy::Final {
        // Sampled producers retain physical coordinates, including their terminal field.
        // Select the authored endpoint without relabeling the field or changing provenance.
        let terminal = sample == RecoveredFieldSample::Final
            || terminal_physical_sample(&plan.independent_domain) == Some(sample);
        return Ok(event_attribution.is_none() && terminal);
    }
    Ok(
        sample_matches_history_policy(&observation.spec.sampling, sample, event_attribution)?
            && sample_matches_domain(&plan.independent_domain, sample),
    )
}

/// Whether any authored observation of `step` samples `sample`: the single rule deciding
/// whether a sampled source (one frequency of a sweep, one time point) carries observations.
pub fn authored_observations_sample(
    step: &runmat_analysis_core::ProcedureStep,
    sample: RecoveredFieldSample,
) -> Result<bool, ObservationError> {
    for observation in &step.observations {
        let Some(spec) = observation.spec() else {
            return Ok(true);
        };
        let accepted = if spec.sampling == SamplingPolicy::Final {
            sample == RecoveredFieldSample::Final
                || terminal_physical_sample(&step.independent_domain) == Some(sample)
        } else {
            sample_matches_history_policy(&spec.sampling, sample, None)?
                && sample_matches_domain(&step.independent_domain, sample)
        };
        if accepted {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether any authored history (an observation sampled over the independent domain, not only
/// at its end) samples `sample`: the rule deciding which interior samples a producer retains.
pub fn authored_histories_sample(
    step: &runmat_analysis_core::ProcedureStep,
    sample: RecoveredFieldSample,
) -> Result<bool, ObservationError> {
    for observation in &step.observations {
        let Some(spec) = observation.spec() else {
            continue;
        };
        if spec.sampling != SamplingPolicy::Final
            && sample_matches_history_policy(&spec.sampling, sample, None)?
            && sample_matches_domain(&step.independent_domain, sample)
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn terminal_physical_sample(
    domain: &runmat_analysis_core::IndependentDomain,
) -> Option<RecoveredFieldSample> {
    match domain {
        runmat_analysis_core::IndependentDomain::TimeSeconds { end, .. } => {
            Some(RecoveredFieldSample::TimeSeconds(*end))
        }
        // Authored frequency domains are validated as nonempty and strictly increasing.
        runmat_analysis_core::IndependentDomain::FrequencyHertz(values) => values
            .last()
            .copied()
            .map(RecoveredFieldSample::FrequencyHertz),
        _ => None,
    }
}

fn evaluate_sample(
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    topology: &ObservationTopology,
    fields: &RecoveredFieldCatalog,
    sample: RecoveredFieldSample,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationResult, ObservationError> {
    let source = fields
        .get(observation.source_field_definition_identity, sample)
        .ok_or_else(|| ObservationError::Field {
            observation_id: observation.spec.observation_id.0.clone(),
            reason: "resolved source field sample was not supplied".into(),
        })?;
    if source.definition().quantity() != observation.spec.quantity
        || source.definition().location() != materialized_field_location(&observation.spec)
        || source.definition().entity_domain() != observation.source_entity_domain
        || source.definition().value_shape() != &observation.spec.value_shape
    {
        return Err(ObservationError::Field {
            observation_id: observation.spec.observation_id.0.clone(),
            reason: "supplied field declaration differs from the resolved plan".into(),
        });
    }
    if source.sample() != sample {
        return Err(ObservationError::Identity(
            "recovered-field sample differs from the requested coordinate",
        ));
    }
    evaluate_one(plan, observation, source, topology, event_attribution)
}

fn validate_plan_targets(
    plan: &ResolvedObservationPlan,
    topology: &ObservationTopology,
) -> Result<(), ObservationError> {
    for observation in &plan.observations {
        match &observation.target {
            ResolvedObservationTarget::Nodes {
                indices,
                identities,
            } => {
                if !indices.iter().zip(identities).all(|(index, identity)| {
                    topology
                        .nodes
                        .get(*index)
                        .is_some_and(|node| node.identity == *identity)
                }) {
                    return Err(ObservationError::Identity(
                        "resolved node target differs from the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Elements {
                domain,
                indices,
                identities,
            } => {
                let elements = topology.elements(*domain)?;
                if !indices.iter().zip(identities).all(|(index, identity)| {
                    elements
                        .get(*index)
                        .is_some_and(|element| element.identity == *identity)
                }) {
                    return Err(ObservationError::Identity(
                        "resolved element target differs from the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Points { samples } => {
                let projection =
                    topology
                        .solid_projection
                        .as_ref()
                        .ok_or(ObservationError::Unsupported(
                            "point observations require admitted solid topology",
                        ))?;
                for sample in samples {
                    let element = projection.elements.get(sample.source_element_index).ok_or(
                        ObservationError::Identity(
                            "resolved point source exceeds the topology projection",
                        ),
                    )?;
                    if element.stable_identity != sample.source_element_identity
                        || element.node_indices != sample.source_node_indices
                        || sample
                            .source_node_indices
                            .iter()
                            .zip(&sample.source_node_identities)
                            .any(|(index, identity)| {
                                projection
                                    .nodes
                                    .get(*index)
                                    .is_none_or(|node| node.stable_identity != *identity)
                            })
                    {
                        return Err(ObservationError::Identity(
                            "resolved point source differs from the topology projection",
                        ));
                    }
                }
            }
            ResolvedObservationTarget::Volume {
                element_indices,
                element_identities,
            } => {
                let projection =
                    topology
                        .solid_projection
                        .as_ref()
                        .ok_or(ObservationError::Unsupported(
                            "volume observations require admitted solid topology",
                        ))?;
                if element_indices
                    .iter()
                    .zip(element_identities)
                    .any(|(index, identity)| {
                        projection
                            .elements
                            .get(*index)
                            .is_none_or(|element| element.stable_identity != *identity)
                    })
                {
                    return Err(ObservationError::Identity(
                        "resolved volume target exceeds the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Boundary {
                face_indices,
                face_identities,
            }
            | ResolvedObservationTarget::BoundaryPair {
                face_indices,
                face_identities,
                ..
            } => {
                let projection =
                    topology
                        .solid_projection
                        .as_ref()
                        .ok_or(ObservationError::Unsupported(
                            "boundary observations require admitted solid topology",
                        ))?;
                if face_indices
                    .iter()
                    .zip(face_identities)
                    .any(|(index, identity)| {
                        projection
                            .boundary_faces
                            .get(*index)
                            .is_none_or(|face| face.stable_identity != *identity)
                    })
                {
                    return Err(ObservationError::Identity(
                        "resolved boundary target exceeds the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::ConstraintGroups { identities, .. } => {
                if identities.len() != 1 || identities[0] == runmat_meshing_core::StableDigest::ZERO
                {
                    return Err(ObservationError::Identity(
                        "resolved constraint target identity is invalid",
                    ));
                }
            }
            ResolvedObservationTarget::Connectors {
                indices,
                identities,
            } => {
                if indices.len() != 1
                    || identities.len() != 1
                    || identities[0] == runmat_meshing_core::StableDigest::ZERO
                {
                    return Err(ObservationError::Identity(
                        "resolved connector target identity is invalid",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn evaluate_one(
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    field: &crate::results::field::RecoveredField,
    topology: &ObservationTopology,
    event_attribution: Option<ObservationEventAttribution>,
) -> Result<ObservationResult, ObservationError> {
    super::distributed::evaluate_partitioned_observation(
        plan,
        observation,
        field,
        topology,
        super::ObservationExecutionPolicy::serial(),
        event_attribution,
    )
}
