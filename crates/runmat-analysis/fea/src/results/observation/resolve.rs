use std::collections::BTreeSet;

use runmat_analysis_core::{
    ComplexRepresentation, CompressionPolicy, EntityDimension, ExcitationTarget, ExpectedTopology,
    FieldLocation, ObservationComponent, ObservationCostEstimate, ObservationRequest,
    ObservationSpec, ObservationTarget, ProcedureStep, RecoveryPolicy, RetentionPolicy,
    SamplingPolicy, SeriesReduction, SignalPrecision, SpatialReduction, TargetCardinality,
};
use runmat_execution::Digest;

use crate::{results::field::RecoveredFieldDefinition, topology::AnalysisProjection};

use super::contract::{
    checked_bytes, shape_component_count, ObservationError, ObservationLimits,
    ResolvedComponentProjection, ResolvedObservation, ResolvedObservationPlan,
    ResolvedObservationPlanIdentity, ResolvedObservationTarget,
};

pub fn resolve_observation_plan(
    step: &ProcedureStep,
    projection: &AnalysisProjection,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservationPlan, ObservationError> {
    if step.observations.len() > limits.maximum_observations {
        return Err(ObservationError::Limit("observation count"));
    }
    validate_field_catalog(fields)?;
    let observations = step
        .observations
        .iter()
        .map(|request| resolve_request(step, request, projection, fields, limits))
        .collect::<Result<Vec<_>, _>>()?;
    let mut plan = ResolvedObservationPlan {
        schema_version: 1,
        identity: ResolvedObservationPlanIdentity(Digest::sha256([])),
        topology_digest: projection.projection_digest,
        step_id: step.step_id.clone(),
        observations,
    };
    plan.identity = plan.derive_identity()?;
    plan.validate_identity()?;
    Ok(plan)
}

fn validate_field_catalog(fields: &[RecoveredFieldDefinition]) -> Result<(), ObservationError> {
    let mut identities = std::collections::BTreeSet::new();
    for field in fields {
        if !identities.insert(field.identity()) {
            return Err(ObservationError::Invalid(
                "recovered-field declarations require unique identities".into(),
            ));
        }
        if !matches!(
            field.location(),
            FieldLocation::Node | FieldLocation::Element
        ) {
            return Err(ObservationError::Unsupported(
                "this evaluator accepts only recovered nodal and element fields",
            ));
        }
        shape_component_count(field.value_shape())?;
    }
    Ok(())
}

fn resolve_request(
    step: &ProcedureStep,
    request: &ObservationRequest,
    projection: &AnalysisProjection,
    fields: &[RecoveredFieldDefinition],
    limits: ObservationLimits,
) -> Result<ResolvedObservation, ObservationError> {
    let spec = match request {
        ObservationRequest::Field(spec)
        | ObservationRequest::Probe(spec)
        | ObservationRequest::Reduction(spec) => spec,
        ObservationRequest::History(_) => {
            return Err(ObservationError::Unsupported(
                "history evaluation requires a sampled state series",
            ));
        }
        ObservationRequest::Diagnostic(_) | ObservationRequest::Monitor(_) => {
            return Err(ObservationError::Unsupported(
                "diagnostics and monitors are evaluated by their owning pipeline",
            ));
        }
    };
    if !runmat_analysis_core::procedure_produces_observation_quantity(step, spec.quantity) {
        return Err(ObservationError::Invalid(
            "observation quantity is not produced by the procedure physics family".into(),
        ));
    }
    validate_supported_spec(request, spec)?;
    let source_field = bind_field(spec, fields)?;
    let target = resolve_target(request, spec, projection)?;
    let (component, result_component_count) = resolve_component(spec, &source_field)?;
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
    let value_count = match request {
        ObservationRequest::Reduction(_) => result_component_count,
        _ => selected
            .checked_mul(result_component_count)
            .ok_or(ObservationError::Limit("observation value count"))?,
    };
    if value_count > limits.maximum_values {
        return Err(ObservationError::Limit("observation value count"));
    }
    let work_entities = integration_work(&target, projection)?;
    let operation_count = u64::try_from(work_entities)
        .ok()
        .and_then(|value| value.checked_mul(result_component_count as u64))
        .and_then(|value| value.checked_mul(operation_multiplier(request, spec)))
        .ok_or(ObservationError::Limit("observation operation count"))?;
    let value_bytes = checked_bytes(value_count)?;
    let source_work_bytes = checked_bytes(
        work_entities
            .checked_mul(shape_component_count(source_field.value_shape())?)
            .ok_or(ObservationError::Limit("observation working-set values"))?,
    )?;
    let stored_bytes = value_bytes
        .checked_add(16 * 1024)
        .ok_or(ObservationError::Limit("observation storage overhead"))?;
    let resolved_cost = ObservationCostEstimate {
        operation_count,
        peak_resident_bytes: source_work_bytes
            .checked_add(value_bytes)
            .ok_or(ObservationError::Limit("observation working set"))?,
        stored_bytes,
        transfer_bytes: stored_bytes,
        collective_count: 0,
    };
    validate_cost(&spec.cost, &resolved_cost, limits)?;
    Ok(ResolvedObservation {
        spec: spec.clone(),
        source_field_definition_identity: source_field.identity(),
        target,
        component,
        result_component_count,
        resolved_cost,
    })
}

fn validate_supported_spec(
    request: &ObservationRequest,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    if spec.source_location != spec.output_location || spec.recovery != RecoveryPolicy::Native {
        return Err(ObservationError::Unsupported(
            "field recovery and location transfer must be completed before observation evaluation",
        ));
    }
    if !matches!(
        spec.source_location,
        FieldLocation::Node | FieldLocation::Element
    ) {
        return Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal and element fields",
        ));
    }
    if spec.complex_representation != ComplexRepresentation::Real
        || spec.storage.precision != SignalPrecision::Real64
    {
        return Err(ObservationError::Unsupported(
            "this serial evaluator currently publishes real f64 observations",
        ));
    }
    if spec.sampling != SamplingPolicy::Final {
        return Err(ObservationError::Unsupported(
            "this evaluator requires one final solver state",
        ));
    }
    if !matches!(
        spec.storage.compression,
        CompressionPolicy::None | CompressionPolicy::Lossless
    ) {
        return Err(ObservationError::Unsupported(
            "error-bounded observation compression is not implemented",
        ));
    }
    match &spec.storage.retention {
        RetentionPolicy::LatestSamples { count } if *count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        RetentionPolicy::ExplicitSamples { maximum_count } if *maximum_count < 1 => {
            return Err(ObservationError::Invalid(
                "retention count must be nonzero".into(),
            ));
        }
        _ => {}
    }
    if let Some(reduction) = &spec.reduction {
        if reduction.series != SeriesReduction::None {
            return Err(ObservationError::Unsupported(
                "series reductions require a sampled state series",
            ));
        }
        if !matches!(
            reduction.spatial,
            SpatialReduction::None
                | SpatialReduction::Integral
                | SpatialReduction::Average
                | SpatialReduction::RootMeanSquare
        ) {
            return Err(ObservationError::Unsupported(
                "this slice supports FE-weighted integral, average, and root-mean-square reductions",
            ));
        }
    }
    match request {
        ObservationRequest::Field(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "field requests cannot apply reductions".into(),
        )),
        ObservationRequest::Probe(_) if spec.reduction.is_some() => Err(ObservationError::Invalid(
            "probe requests cannot apply reductions".into(),
        )),
        ObservationRequest::Reduction(_)
            if spec
                .reduction
                .as_ref()
                .is_none_or(|reduction| reduction.spatial == SpatialReduction::None) =>
        {
            Err(ObservationError::Invalid(
                "reduction requests require a spatial reduction".into(),
            ))
        }
        _ => Ok(()),
    }
}

fn bind_field(
    spec: &ObservationSpec,
    fields: &[RecoveredFieldDefinition],
) -> Result<RecoveredFieldDefinition, ObservationError> {
    let matches = fields
        .iter()
        .filter(|field| {
            field.quantity() == spec.quantity
                && field.location() == spec.source_location
                && field.value_shape() == &spec.value_shape
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [field] => Ok((*field).clone()),
        [] => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason: "no declaration matches quantity, location, and value shape".into(),
        }),
        _ => Err(ObservationError::Field {
            observation_id: spec.observation_id.0.clone(),
            reason:
                "multiple declarations match; recovery must provide an unambiguous field catalog"
                    .into(),
        }),
    }
}

fn resolve_target(
    request: &ObservationRequest,
    spec: &ObservationSpec,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    match request {
        ObservationRequest::Probe(_) => resolve_probe(spec, projection),
        ObservationRequest::Reduction(_) => resolve_reduction_target(spec, projection),
        ObservationRequest::Field(_) => resolve_field_target(spec, projection),
        _ => Err(ObservationError::Unsupported(
            "unsupported observation request",
        )),
    }
}

fn resolve_probe(
    spec: &ObservationSpec,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let ObservationTarget::Point { position } = spec.target else {
        return Err(target_error(
            spec,
            "probe requests require an exact point target",
        ));
    };
    match spec.source_location {
        FieldLocation::Node => {
            let matches = projection
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.coordinates_m == position.meters)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [(index, node)] => Ok(ResolvedObservationTarget::Nodes {
                    indices: vec![*index],
                    identities: vec![node.stable_identity],
                }),
                [] => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted mesh node",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple admitted mesh nodes",
                )),
            }
        }
        FieldLocation::Element => {
            let matches = projection
                .elements
                .iter()
                .enumerate()
                .filter(|(_, element)| element.centroid_m == position.meters)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [(index, element)] => Ok(ResolvedObservationTarget::Elements {
                    indices: vec![*index],
                    identities: vec![element.stable_identity],
                }),
                [] => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted element centroid",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple element centroids",
                )),
            }
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal and element probes",
        )),
    }
}

fn resolve_field_target(
    spec: &ObservationSpec,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => match spec.source_location {
            FieldLocation::Node => nodes_target((0..projection.nodes.len()).collect(), projection),
            FieldLocation::Element => {
                elements_target((0..projection.elements.len()).collect(), projection)
            }
            _ => Err(ObservationError::Unsupported("unsupported field location")),
        },
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (
                region.entity_dimension,
                region.expected_topology,
                spec.source_location,
            ) {
                (EntityDimension::Point, ExpectedTopology::MeshNode, FieldLocation::Node) => {
                    let id = parse_node_selector(&region.region_id, spec)?;
                    let index = projection
                        .node_index(id)
                        .ok_or_else(|| target_error(spec, "mesh-node selector does not resolve"))?;
                    enforce_cardinality(region.cardinality, 1, spec)?;
                    nodes_target(vec![index], projection)
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace, FieldLocation::Node) => {
                    let faces = matching_faces(&region.region_id, projection);
                    enforce_cardinality(region.cardinality, faces.len(), spec)?;
                    let indices = faces
                        .into_iter()
                        .flat_map(|index| {
                            projection.boundary_faces[index]
                                .node_indices
                                .iter()
                                .copied()
                        })
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    nodes_target(indices, projection)
                }
                (
                    EntityDimension::Volume,
                    ExpectedTopology::MeshElement,
                    FieldLocation::Element,
                ) => {
                    let indices = matching_elements(&region.region_id, projection);
                    enforce_cardinality(region.cardinality, indices.len(), spec)?;
                    elements_target(indices, projection)
                }
                (EntityDimension::Volume, ExpectedTopology::MeshElement, FieldLocation::Node) => {
                    let elements = matching_elements(&region.region_id, projection);
                    enforce_cardinality(region.cardinality, elements.len(), spec)?;
                    let indices = elements
                        .into_iter()
                        .flat_map(|index| projection.elements[index].node_indices.iter().copied())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    nodes_target(indices, projection)
                }
                _ => Err(target_error(
                    spec,
                    "target topology is incompatible with the recovered field location",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "field requests support whole-model and admitted region targets",
        )),
    }
}

fn resolve_reduction_target(
    spec: &ObservationSpec,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
            Ok(ResolvedObservationTarget::Volume {
                element_indices: (0..projection.elements.len()).collect(),
            })
        }
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (region.entity_dimension, region.expected_topology) {
                (EntityDimension::Volume, ExpectedTopology::MeshElement) => {
                    let element_indices = matching_elements(&region.region_id, projection);
                    enforce_cardinality(region.cardinality, element_indices.len(), spec)?;
                    Ok(ResolvedObservationTarget::Volume { element_indices })
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace)
                    if spec.source_location == FieldLocation::Node =>
                {
                    let face_indices = matching_faces(&region.region_id, projection);
                    enforce_cardinality(region.cardinality, face_indices.len(), spec)?;
                    Ok(ResolvedObservationTarget::Boundary { face_indices })
                }
                _ => Err(target_error(
                    spec,
                    "reductions support admitted volumes and nodal fields on admitted boundaries",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "reductions support whole-model and admitted region targets",
        )),
    }
}

fn nodes_target(
    indices: Vec<usize>,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let identities = indices
        .iter()
        .map(|index| projection.nodes[*index].stable_identity)
        .collect();
    Ok(ResolvedObservationTarget::Nodes {
        indices,
        identities,
    })
}

fn elements_target(
    indices: Vec<usize>,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let identities = indices
        .iter()
        .map(|index| projection.elements[*index].stable_identity)
        .collect();
    Ok(ResolvedObservationTarget::Elements {
        indices,
        identities,
    })
}

fn matching_elements(selector: &str, projection: &AnalysisProjection) -> Vec<usize> {
    projection
        .elements
        .iter()
        .enumerate()
        .filter_map(|(index, element)| {
            element
                .region_id
                .matches_analysis_selection(selector)
                .then_some(index)
        })
        .collect()
}

fn matching_faces(selector: &str, projection: &AnalysisProjection) -> Vec<usize> {
    projection
        .boundary_faces
        .iter()
        .enumerate()
        .filter_map(|(index, face)| {
            face.provenance
                .iter()
                .any(|entity| entity.matches_analysis_selection(selector))
                .then_some(index)
        })
        .collect()
}

fn parse_node_selector(selector: &str, spec: &ObservationSpec) -> Result<u64, ObservationError> {
    selector
        .strip_prefix("node:")
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| target_error(spec, "mesh-node targets require a `node:<id>` selector"))
}

fn enforce_cardinality(
    cardinality: TargetCardinality,
    count: usize,
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    if count == 0 {
        Err(target_error(
            spec,
            "region resolves to no admitted topology",
        ))
    } else if cardinality == TargetCardinality::One && count != 1 {
        Err(target_error(
            spec,
            "region cardinality requires exactly one matched entity",
        ))
    } else {
        Ok(())
    }
}

fn resolve_component(
    spec: &ObservationSpec,
    field: &RecoveredFieldDefinition,
) -> Result<(ResolvedComponentProjection, usize), ObservationError> {
    let source = shape_component_count(field.value_shape())?;
    match spec.component {
        ObservationComponent::All => Ok((ResolvedComponentProjection::All, source)),
        ObservationComponent::X if source == 3 => {
            Ok((ResolvedComponentProjection::Cartesian { axis: 0 }, 1))
        }
        ObservationComponent::Y if source == 3 => {
            Ok((ResolvedComponentProjection::Cartesian { axis: 1 }, 1))
        }
        ObservationComponent::Z if source == 3 => {
            Ok((ResolvedComponentProjection::Cartesian { axis: 2 }, 1))
        }
        ObservationComponent::Magnitude if source == 3 => {
            Ok((ResolvedComponentProjection::Magnitude3, 1))
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator supports scalar/all and vector all, Cartesian, or magnitude components",
        )),
    }
}

fn integration_work(
    target: &ResolvedObservationTarget,
    projection: &AnalysisProjection,
) -> Result<usize, ObservationError> {
    match target {
        ResolvedObservationTarget::Nodes { indices, .. }
        | ResolvedObservationTarget::Elements { indices, .. } => Ok(indices.len()),
        ResolvedObservationTarget::Volume { element_indices } => {
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
        ResolvedObservationTarget::Boundary { face_indices } => {
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
    match request {
        ObservationRequest::Reduction(_) => {
            match spec.reduction.as_ref().map(|reduction| reduction.spatial) {
                Some(SpatialReduction::RootMeanSquare) => 12,
                Some(SpatialReduction::Integral | SpatialReduction::Average) => 8,
                _ => 3,
            }
        }
        _ => 2,
    }
}

fn validate_cost(
    authored: &ObservationCostEstimate,
    resolved: &ObservationCostEstimate,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
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
            "authored observation cost estimate understates the resolved serial plan".into(),
        ));
    }
    if authored.operation_count > limits.maximum_operation_count
        || resolved.operation_count > limits.maximum_operation_count
    {
        return Err(ObservationError::Limit("operation count"));
    }
    if authored.peak_resident_bytes > limits.maximum_peak_resident_bytes
        || resolved.peak_resident_bytes > limits.maximum_peak_resident_bytes
    {
        return Err(ObservationError::Limit("peak resident bytes"));
    }
    if authored.stored_bytes > limits.maximum_stored_bytes
        || resolved.stored_bytes > limits.maximum_stored_bytes
    {
        return Err(ObservationError::Limit("stored bytes"));
    }
    if authored.transfer_bytes > limits.maximum_transfer_bytes
        || resolved.transfer_bytes > limits.maximum_transfer_bytes
    {
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
