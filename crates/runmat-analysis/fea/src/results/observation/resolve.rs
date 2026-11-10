use runmat_analysis_core::{
    AnalysisModel, ComplexRepresentation, CompressionPolicy, EntityDimension, ExcitationTarget,
    ExpectedTopology, FieldLocation, ObservationComponent, ObservationCostEstimate,
    ObservationQuantity, ObservationRequest, ObservationSpec, ObservationTarget,
    ObservationValueShape, ProcedureStep, RecoveryPolicy, RetentionPolicy, SamplingPolicy,
    SeriesReduction, SignConvention, SpatialReduction, TargetCardinality,
};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};

use crate::{
    connectors::CanonicalStructuralConnectorPlan,
    constraints::CanonicalStructuralConstraintPlan,
    results::field::{
        RecoveredFieldDefinition, RecoveredFieldDerivation, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation,
    },
    topology::AnalysisProjection,
};

use super::contract::{
    checked_bytes, derive_resolved_point_identity, materialized_field_location,
    observation_result_storage_component_count, shape_component_count,
    validate_output_numeric_contract, ObservationError, ObservationLimits, ObservationSampleBounds,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanContents, ResolvedObservationPlanIdentity, ResolvedObservationPoint,
    ResolvedObservationTarget,
};
use super::history::resolve_history_sample_bounds;
use super::topology::ObservationTopology;

mod projection;
pub use projection::resolve_observation_plan;

mod field_catalog;
use field_catalog::{bind_field, validate_field_catalog};

mod connectors;
use connectors::{compile_connector_observation_plan, resolve_connector_target};
mod targets;
use targets::resolve_target;
mod validation;
mod verification;
use validation::{
    validate_numeric_representation, validate_resolved_spatial_reduction, validate_supported_spec,
};
pub(crate) use verification::{
    verify_authored_observations, verify_observation_plan, verify_sampled_observation_plan,
    verify_structural_observation_plan,
};

pub fn resolve_structural_observation_plan(
    step: &ProcedureStep,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    validate_request_counts(step, limits)?;
    let topology = ObservationTopology::derive(model, mesh)?;
    let connector_plan =
        compile_connector_observation_plan(model, mesh, constraint_plan, topology.nodes.len())?;
    validate_structural_context(model, &topology, constraint_plan, &connector_plan)?;
    resolve_observation_plan_for_topology(
        step,
        &topology,
        Some(constraint_plan),
        Some(&connector_plan),
        fields,
        limits,
    )
}

pub fn resolve_structural_observation_plan_with_connectors(
    step: &ProcedureStep,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    connector_plan: &CanonicalStructuralConnectorPlan,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    validate_request_counts(step, limits)?;
    let topology = ObservationTopology::derive(model, mesh)?;
    validate_structural_context(model, &topology, constraint_plan, connector_plan)?;
    resolve_observation_plan_for_topology(
        step,
        &topology,
        Some(constraint_plan),
        Some(connector_plan),
        fields,
        limits,
    )
}

fn validate_structural_context(
    model: &AnalysisModel,
    topology: &ObservationTopology,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    connector_plan: &CanonicalStructuralConnectorPlan,
) -> Result<(), ObservationError> {
    constraint_plan
        .validate()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    if Some(constraint_plan.topology) != topology.identity {
        return Err(ObservationError::Identity(
            "constraint plan topology differs from the observation topology",
        ));
    }
    connector_plan
        .validate()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    connector_plan
        .validate_against_authored(&model.connectors)
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    if Some(connector_plan.topology) != topology.identity {
        return Err(ObservationError::Identity(
            "connector plan topology differs from the observation topology",
        ));
    }
    Ok(())
}

pub(super) fn validate_request_counts(
    step: &ProcedureStep,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
    if step.observations.len() > limits.maximum_observations {
        return Err(ObservationError::Limit("observation count"));
    }
    for request in &step.observations {
        let spec = match request {
            ObservationRequest::Field(spec)
            | ObservationRequest::History(spec)
            | ObservationRequest::Probe(spec)
            | ObservationRequest::Reduction(spec) => spec,
            _ => continue,
        };
        if let ObservationTarget::Path { positions } = &spec.target {
            if positions.len() > limits.maximum_selected_entities {
                return Err(ObservationError::Limit("selected topology entities"));
            }
        }
    }
    Ok(())
}

pub(super) fn resolve_observation_plan_for_topology(
    step: &ProcedureStep,
    topology: &ObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    connector_plan: Option<&CanonicalStructuralConnectorPlan>,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    if step.observations.len() > limits.maximum_observations {
        return Err(ObservationError::Limit("observation count"));
    }
    validate_field_catalog(fields.iter(), limits.maximum_peak_resident_bytes)?;
    let observations = step
        .observations
        .iter()
        .map(|request| {
            resolve_request(
                step,
                request,
                topology,
                constraint_plan,
                connector_plan,
                fields.iter(),
                limits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    ResolvedObservationPlan::seal_derived(ResolvedObservationPlanContents {
        schema_version: 1,
        identity: ResolvedObservationPlanIdentity(Digest::sha256([])),
        topology_digest: topology.digest(),
        step_id: step.step_id.clone(),
        independent_domain: step.independent_domain.clone(),
        observations,
    })
}

/// The mesh-free part of resolving one authored request: sampling, the family that produces
/// the quantity, the supported spec shape, and the one declared field it binds to with a
/// matching numeric representation. Static admission (`runmat check` and run preparation) and
/// plan resolution both call this, so they accept the same observations. Diagnostics and
/// monitors bind to their owning pipeline and return `None`.
pub(crate) fn bind_request_field<'r, 'a>(
    step: &ProcedureStep,
    request: &'r ObservationRequest,
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition> + Clone,
) -> Result<Option<(&'r ObservationSpec, &'a RecoveredFieldDefinition)>, ObservationError> {
    runmat_analysis_core::validate_observation_sampling(step, request)
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let spec = match request {
        ObservationRequest::Field(spec)
        | ObservationRequest::History(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec) => spec,
        ObservationRequest::Diagnostic(_) | ObservationRequest::Monitor(_) => return Ok(None),
    };
    if !runmat_analysis_core::procedure_produces_observation_quantity(step, spec.quantity) {
        return Err(ObservationError::Invalid(
            "observation quantity is not produced by the procedure physics family".into(),
        ));
    }
    validate_supported_spec(request, spec)?;
    let source_field = bind_field(spec, fields)?;
    validate_numeric_representation(spec, source_field)?;
    Ok(Some((spec, source_field)))
}

fn resolve_request<'a>(
    step: &ProcedureStep,
    request: &ObservationRequest,
    topology: &ObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    connector_plan: Option<&CanonicalStructuralConnectorPlan>,
    fields: impl Iterator<Item = &'a RecoveredFieldDefinition> + Clone,
    limits: ObservationLimits,
) -> Result<ResolvedObservation, ObservationError> {
    let Some((spec, source_field)) = bind_request_field(step, request, fields)? else {
        return Err(ObservationError::Unsupported(
            "diagnostics and monitors are evaluated by their owning pipeline",
        ));
    };
    let target = resolve_target(
        request,
        spec,
        source_field.entity_domain(),
        topology,
        constraint_plan,
        connector_plan,
        limits,
    )?;
    let target = restrict_to_support(target, source_field, topology)?;
    validate_resolved_spatial_reduction(spec, source_field, &target)?;
    let component = resolve_component(spec, source_field)?;
    let result_value_shape = spec.result_value_shape();
    let result_component_count = shape_component_count(&result_value_shape)?;
    let spatial_extrema = spec.reduction.as_ref().is_some_and(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
    });
    let has_spatial_reduction = spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    if spatial_extrema && result_component_count != 1 {
        return Err(ObservationError::Invalid(
            "spatial extrema require a scalar quantity or scalar component projection".into(),
        ));
    }
    let selected = target.selected_entity_count();
    if selected == 0 {
        return Err(target_error(
            spec,
            "target resolves to no admitted topology",
        ));
    }
    if selected > limits.maximum_selected_entities {
        return Err(ObservationError::Limit("selected topology entities"));
    }
    let result_storage_component_count = observation_result_storage_component_count(
        result_component_count,
        spec.complex_representation,
    )?;
    let value_count = match request {
        _ if spatial_extrema
            && spec.reduction.as_ref().is_some_and(|reduction| {
                reduction.extrema_ties == runmat_analysis_core::ExtremaTiePolicy::AllLocations
            }) =>
        {
            selected
                .checked_mul(result_storage_component_count)
                .ok_or(ObservationError::Limit("observation extrema value count"))?
        }
        ObservationRequest::Reduction(_) => result_storage_component_count,
        ObservationRequest::History(_) if has_spatial_reduction => result_storage_component_count,
        _ => selected
            .checked_mul(result_storage_component_count)
            .ok_or(ObservationError::Limit("observation value count"))?,
    };
    if value_count > limits.maximum_values {
        return Err(ObservationError::Limit("observation value count"));
    }
    let sample_bounds = match request {
        ObservationRequest::History(_) => resolve_history_sample_bounds(
            &spec.sampling,
            &spec.storage.retention,
            step.accuracy.resources.maximum_samples,
        )?,
        _ => ObservationSampleBounds {
            maximum_produced_samples: 1,
            maximum_retained_samples: 1,
        },
    };
    if spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.series != SeriesReduction::None)
        && sample_bounds.maximum_retained_samples < sample_bounds.maximum_produced_samples
    {
        return Err(ObservationError::Invalid(
            "series reduction requires retention of every produced history sample".into(),
        ));
    }
    let retained_sample_count = u64::try_from(sample_bounds.maximum_retained_samples)
        .map_err(|_| ObservationError::Limit("observation retained sample count"))?;
    let result_component_count_u64 = u64::try_from(result_storage_component_count)
        .map_err(|_| ObservationError::Limit("observation result component count"))?;
    let work_entities = integration_work(&target, topology)?;
    let operation_count = u64::try_from(work_entities)
        .ok()
        .and_then(|value| value.checked_mul(result_component_count_u64))
        .and_then(|value| value.checked_mul(operation_multiplier(request, spec)))
        .and_then(|value| {
            value.checked_mul(u64::try_from(sample_bounds.maximum_produced_samples).ok()?)
        })
        .ok_or(ObservationError::Limit("observation operation count"))?;
    let value_bytes = checked_bytes(value_count)?;
    let retained_value_bytes = value_bytes
        .checked_mul(retained_sample_count)
        .ok_or(ObservationError::Limit("observation retained value bytes"))?;
    let series_output_bytes = match spec.reduction.as_ref().map(|reduction| reduction.series) {
        Some(SeriesReduction::Envelope) => value_bytes
            .checked_mul(2)
            .ok_or(ObservationError::Limit("series envelope bytes"))?,
        Some(series) if series != SeriesReduction::None => value_bytes,
        _ => 0,
    };
    let series_evidence_bytes = if series_output_bytes == 0 {
        0
    } else {
        retained_value_bytes
    };
    let source_work_bytes = checked_bytes(
        work_entities
            .checked_mul(
                source_field
                    .storage_component_count()
                    .map_err(|error| ObservationError::Invalid(error.to_string()))?,
            )
            .ok_or(ObservationError::Limit("observation working-set values"))?,
    )?;
    let stored_bytes = retained_value_bytes
        .checked_add(
            16_u64
                .checked_mul(1024)
                .and_then(|overhead| overhead.checked_mul(retained_sample_count))
                .ok_or(ObservationError::Limit("observation storage overhead"))?,
        )
        .and_then(|value| value.checked_add(series_output_bytes))
        .and_then(|value| value.checked_add(series_evidence_bytes))
        .ok_or(ObservationError::Limit("observation storage overhead"))?;
    let resolved_cost = ObservationCostEstimate {
        operation_count,
        peak_resident_bytes: source_work_bytes
            .checked_add(retained_value_bytes)
            .and_then(|value| value.checked_add(series_output_bytes))
            .and_then(|value| value.checked_add(series_evidence_bytes))
            .ok_or(ObservationError::Limit("observation working set"))?,
        stored_bytes,
        transfer_bytes: stored_bytes,
        collective_count: 0,
    };
    validate_cost(spec.cost, &resolved_cost, limits)?;
    Ok(ResolvedObservation {
        spec: spec.clone(),
        source_field_definition_identity: source_field.identity(),
        source_entity_domain: source_field.entity_domain(),
        target,
        component,
        result_value_shape,
        sample_bounds,
        resolved_cost,
    })
}

fn resolve_component(
    spec: &ObservationSpec,
    field: &RecoveredFieldDefinition,
) -> Result<ResolvedComponentProjection, ObservationError> {
    let source = shape_component_count(field.value_shape())?;
    match spec.component {
        ObservationComponent::All => Ok(ResolvedComponentProjection::All),
        ObservationComponent::X if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 0 })
        }
        ObservationComponent::Y if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 1 })
        }
        ObservationComponent::Z if source == 3 => {
            Ok(ResolvedComponentProjection::Cartesian { axis: 2 })
        }
        ObservationComponent::Magnitude if source == 3 => {
            Ok(ResolvedComponentProjection::Magnitude3)
        }
        ObservationComponent::VonMises if source == 6 => {
            Ok(ResolvedComponentProjection::VonMisesSymmetricTensor)
        }
        ObservationComponent::MaximumPrincipal if source == 6 => {
            Ok(ResolvedComponentProjection::MaximumPrincipalSymmetricTensor)
        }
        ObservationComponent::MinimumPrincipal if source == 6 => {
            Ok(ResolvedComponentProjection::MinimumPrincipalSymmetricTensor)
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator does not support the requested component for the recovered value shape",
        )),
    }
}

fn integration_work(
    target: &ResolvedObservationTarget,
    topology: &ObservationTopology,
) -> Result<usize, ObservationError> {
    match target {
        ResolvedObservationTarget::Nodes { indices, .. }
        | ResolvedObservationTarget::Elements { indices, .. }
        | ResolvedObservationTarget::Connectors { indices, .. }
        | ResolvedObservationTarget::ConstraintGroups { indices, .. } => Ok(indices.len()),
        ResolvedObservationTarget::Points { samples } => {
            samples.iter().try_fold(0usize, |work, sample| {
                work.checked_add(sample.source_node_indices.len().saturating_mul(8))
                    .ok_or(ObservationError::Limit("point interpolation work"))
            })
        }
        ResolvedObservationTarget::Volume {
            element_indices, ..
        } => {
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "volume reduction work requires admitted solid topology",
                    ))?;
            element_indices.iter().try_fold(0usize, |work, index| {
                projection
                    .elements
                    .get(*index)
                    .and_then(|element| {
                        work.checked_add(element.node_indices.len().saturating_mul(8))
                    })
                    .ok_or(ObservationError::Limit("volume reduction work"))
            })
        }
        ResolvedObservationTarget::Boundary { face_indices, .. }
        | ResolvedObservationTarget::BoundaryPair { face_indices, .. } => {
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "boundary reduction work requires admitted solid topology",
                    ))?;
            face_indices.iter().try_fold(0usize, |work, index| {
                projection
                    .boundary_faces
                    .get(*index)
                    .and_then(|face| work.checked_add(face.node_indices.len().saturating_mul(8)))
                    .ok_or(ObservationError::Limit("boundary reduction work"))
            })
        }
    }
}

fn operation_multiplier(request: &ObservationRequest, spec: &ObservationSpec) -> u64 {
    if spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.series != SeriesReduction::None)
    {
        return 16;
    }
    match request {
        ObservationRequest::Reduction(_)
        | ObservationRequest::History(ObservationSpec {
            reduction: Some(_), ..
        }) => match spec.reduction.as_ref().map(|reduction| reduction.spatial) {
            Some(SpatialReduction::RootMeanSquare) => 12,
            Some(
                SpatialReduction::Integral
                | SpatialReduction::Average
                | SpatialReduction::NormalFlux,
            ) => 8,
            _ => 3,
        },
        _ => 2,
    }
}

fn validate_cost(
    authored: runmat_analysis_core::ObservationCostPolicy,
    resolved: &ObservationCostEstimate,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
    if let Some(authored) = authored.upper_bound() {
        if authored.collective_count != 0 {
            return Err(ObservationError::Unsupported(
                "serial observation evaluation cannot promise collective operations",
            ));
        }
        if authored.operation_count < resolved.operation_count
            || authored.peak_resident_bytes < resolved.peak_resident_bytes
            || authored.stored_bytes < resolved.stored_bytes
            || authored.transfer_bytes < resolved.transfer_bytes
        {
            return Err(ObservationError::Invalid(
                "observation cost upper bound understates the resolved serial plan".into(),
            ));
        }
        if authored.operation_count > limits.maximum_operation_count
            || authored.peak_resident_bytes > limits.maximum_peak_resident_bytes
            || authored.stored_bytes > limits.maximum_stored_bytes
            || authored.transfer_bytes > limits.maximum_transfer_bytes
        {
            return Err(ObservationError::Limit("observation cost upper bound"));
        }
    }
    if resolved.operation_count > limits.maximum_operation_count {
        return Err(ObservationError::Limit("operation count"));
    }
    if resolved.peak_resident_bytes > limits.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit("peak resident bytes"));
    }
    if resolved.stored_bytes > limits.maximum_stored_bytes {
        return Err(ObservationError::Limit("stored bytes"));
    }
    if resolved.transfer_bytes > limits.maximum_transfer_bytes {
        return Err(ObservationError::Limit("transfer bytes"));
    }
    Ok(())
}

fn target_error(spec: &ObservationSpec, reason: impl Into<String>) -> ObservationError {
    ObservationError::Target {
        observation_id: spec.observation_id.0.clone(),
        reason: reason.into(),
    }
}

/// A region-supported field is observed only where it carries values: volume, element and node
/// targets keep their entities inside the support, and an empty intersection fails closed.
fn restrict_to_support(
    target: ResolvedObservationTarget,
    field: &RecoveredFieldDefinition,
    topology: &ObservationTopology,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if field.support().is_complete() {
        return Ok(target);
    }
    let support_topology = topology
        .support_topology()
        .ok_or(ObservationError::Identity(
            "a region-supported field needs a mesh projection topology",
        ))?;
    let resolve = |domain| {
        field
            .support()
            .resolve(domain, support_topology)
            .map_err(|error| ObservationError::Invalid(error.to_string()))
            .map(|support| support.unwrap_or_default())
    };
    let keep = |indices: Vec<usize>, identities: Vec<StableDigest>, support: &[usize]| {
        indices
            .into_iter()
            .zip(identities)
            .filter(|(index, _)| support.binary_search(index).is_ok())
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };
    let restricted = match target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => {
            let support = resolve(crate::results::field::RecoveredFieldEntityDomain::Nodes)?;
            let (indices, identities) = keep(indices, identities, &support);
            ResolvedObservationTarget::Nodes {
                indices,
                identities,
            }
        }
        ResolvedObservationTarget::Elements {
            domain,
            indices,
            identities,
        } => {
            let support = resolve(domain)?;
            let (indices, identities) = keep(indices, identities, &support);
            ResolvedObservationTarget::Elements {
                domain,
                indices,
                identities,
            }
        }
        ResolvedObservationTarget::Volume {
            element_indices,
            element_identities,
        } => {
            let support =
                resolve(crate::results::field::RecoveredFieldEntityDomain::VolumeElements)?;
            let (element_indices, element_identities) =
                keep(element_indices, element_identities, &support);
            ResolvedObservationTarget::Volume {
                element_indices,
                element_identities,
            }
        }
        ResolvedObservationTarget::Boundary {
            face_indices,
            face_identities,
        } if field.entity_domain()
            == crate::results::field::RecoveredFieldEntityDomain::OrientedFaces =>
        {
            let support =
                resolve(crate::results::field::RecoveredFieldEntityDomain::OrientedFaces)?;
            let (face_indices, face_identities) = keep(face_indices, face_identities, &support);
            ResolvedObservationTarget::Boundary {
                face_indices,
                face_identities,
            }
        }
        other => other,
    };
    if restricted.selected_entity_count() == 0 {
        return Err(ObservationError::Invalid(
            "observation target lies entirely outside the field's supported regions".into(),
        ));
    }
    Ok(restricted)
}
