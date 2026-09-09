use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationId, ObservationValueShape, SamplingPolicy,
    SpatialReduction,
};
use runmat_execution::Digest;

use crate::{
    results::field::{
        RecoveredFieldCatalog, RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
    topology::AnalysisProjection,
};

use super::contract::{
    shape_component_count, ObservationEntity, ObservationError, ObservationResult,
    ObservationResultIdentity, ResolvedComponentProjection, ResolvedObservation,
    ResolvedObservationPlan, ResolvedObservationTarget,
};
use super::history::{sample_matches_domain, sample_matches_policy};

pub fn evaluate_observations(
    plan: &ResolvedObservationPlan,
    projection: &AnalysisProjection,
    fields: &RecoveredFieldCatalog,
) -> Result<Vec<ObservationResult>, ObservationError> {
    plan.validate_identity()?;
    if plan.topology_digest != projection.projection_digest {
        return Err(ObservationError::Identity(
            "observation plan names a different topology projection",
        ));
    }
    if fields.topology_digest() != projection.projection_digest {
        return Err(ObservationError::Identity(
            "recovered field catalog names a different topology projection",
        ));
    }
    validate_plan_targets(plan, projection)?;
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
            evaluate_sample(
                plan,
                observation,
                projection,
                fields,
                RecoveredFieldSample::Final,
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
    plan.validate_identity()?;
    if plan.topology_digest != projection.projection_digest
        || fields.topology_digest() != projection.projection_digest
    {
        return Err(ObservationError::Identity(
            "sampled observation inputs name different topology projections",
        ));
    }
    validate_plan_targets(plan, projection)?;
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
    if !sample_matches_policy(&observation.spec.sampling, sample)? {
        return Err(ObservationError::Invalid(
            "sample coordinate differs from the resolved observation schedule".into(),
        ));
    }
    if !sample_matches_domain(&plan.independent_domain, sample) {
        return Err(ObservationError::Invalid(
            "sample coordinate differs from the resolved independent domain".into(),
        ));
    }
    evaluate_sample(plan, observation, projection, fields, sample)
}

fn evaluate_sample(
    plan: &ResolvedObservationPlan,
    observation: &ResolvedObservation,
    projection: &AnalysisProjection,
    fields: &RecoveredFieldCatalog,
    sample: RecoveredFieldSample,
) -> Result<ObservationResult, ObservationError> {
    let source = fields
        .get(observation.source_field_definition_identity, sample)
        .ok_or_else(|| ObservationError::Field {
            observation_id: observation.spec.observation_id.0.clone(),
            reason: "resolved source field sample was not supplied".into(),
        })?;
    if source.definition().quantity() != observation.spec.quantity
        || source.definition().location() != observation.spec.source_location
        || source.definition().value_shape() != &observation.spec.value_shape
    {
        return Err(ObservationError::Field {
            observation_id: observation.spec.observation_id.0.clone(),
            reason: "supplied field declaration differs from the resolved plan".into(),
        });
    }
    if source.sample() != sample
        || source.definition().numeric_representation() != RecoveredFieldNumericRepresentation::Real
    {
        return Err(ObservationError::Unsupported(
            "this serial evaluator requires a matching real recovered-field sample",
        ));
    }
    evaluate_one(plan, observation, source.storage(), projection)
}

fn validate_plan_targets(
    plan: &ResolvedObservationPlan,
    projection: &AnalysisProjection,
) -> Result<(), ObservationError> {
    for observation in &plan.observations {
        match &observation.target {
            ResolvedObservationTarget::Nodes {
                indices,
                identities,
            } => {
                if !indices.iter().zip(identities).all(|(index, identity)| {
                    projection
                        .nodes
                        .get(*index)
                        .is_some_and(|node| node.stable_identity == *identity)
                }) {
                    return Err(ObservationError::Identity(
                        "resolved node target differs from the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Elements {
                indices,
                identities,
            } => {
                if !indices.iter().zip(identities).all(|(index, identity)| {
                    projection
                        .elements
                        .get(*index)
                        .is_some_and(|element| element.stable_identity == *identity)
                }) {
                    return Err(ObservationError::Identity(
                        "resolved element target differs from the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Volume { element_indices } => {
                if element_indices
                    .iter()
                    .any(|index| projection.elements.get(*index).is_none())
                {
                    return Err(ObservationError::Identity(
                        "resolved volume target exceeds the topology projection",
                    ));
                }
            }
            ResolvedObservationTarget::Boundary { face_indices } => {
                if face_indices
                    .iter()
                    .any(|index| projection.boundary_faces.get(*index).is_none())
                {
                    return Err(ObservationError::Identity(
                        "resolved boundary target exceeds the topology projection",
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
    field: &AnalysisField,
    projection: &AnalysisProjection,
) -> Result<ObservationResult, ObservationError> {
    let source_components = shape_component_count(&observation.spec.value_shape)?;
    let source_values = field.as_host_f64().ok_or(ObservationError::Unsupported(
        "serial observation evaluation requires host-resident f64 fields",
    ))?;
    let (values, entities) = match &observation.target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => (
            select_values(
                source_values,
                indices,
                source_components,
                observation.component,
            )?,
            identities
                .iter()
                .copied()
                .map(ObservationEntity::Node)
                .collect(),
        ),
        ResolvedObservationTarget::Elements {
            indices,
            identities,
        } => (
            select_values(
                source_values,
                indices,
                source_components,
                observation.component,
            )?,
            identities
                .iter()
                .copied()
                .map(ObservationEntity::Element)
                .collect(),
        ),
        ResolvedObservationTarget::Volume { element_indices } => (
            reduce_volume(
                observation,
                source_values,
                source_components,
                element_indices,
                projection,
            )?,
            Vec::new(),
        ),
        ResolvedObservationTarget::Boundary { face_indices } => (
            reduce_boundary(
                observation,
                source_values,
                source_components,
                face_indices,
                projection,
            )?,
            Vec::new(),
        ),
    };
    let reduced = matches!(
        observation.target,
        ResolvedObservationTarget::Volume { .. } | ResolvedObservationTarget::Boundary { .. }
    );
    let shape = result_shape(values.len(), observation.result_component_count, reduced);
    let value_shape = if observation.result_component_count == 1 {
        ObservationValueShape::Scalar
    } else {
        observation.spec.value_shape.clone()
    };
    let mut result = ObservationResult {
        schema_version: 1,
        logical_identity: ObservationResultIdentity(Digest::sha256([])),
        plan_identity: plan.identity,
        topology_digest: projection.projection_digest,
        observation_id: observation.spec.observation_id.clone(),
        quantity: observation.spec.quantity,
        location: observation.spec.output_location,
        value_shape,
        entities,
        field: AnalysisField::host_f64(observation.spec.observation_id.0.clone(), shape, values),
        partial_access: observation.spec.storage.partial_access,
    };
    result.logical_identity = result.derive_identity()?;
    result.validate()?;
    Ok(result)
}

fn result_shape(value_count: usize, components: usize, reduced: bool) -> Vec<usize> {
    if reduced || components == 1 {
        vec![value_count]
    } else {
        vec![value_count / components, components]
    }
}

fn select_values(
    values: &[f64],
    indices: &[usize],
    components: usize,
    component: ResolvedComponentProjection,
) -> Result<Vec<f64>, ObservationError> {
    let mut selected = Vec::with_capacity(indices.len().saturating_mul(components));
    for &index in indices {
        let start = index
            .checked_mul(components)
            .ok_or(ObservationError::Limit("field offset"))?;
        let end = start
            .checked_add(components)
            .ok_or(ObservationError::Limit("field offset"))?;
        let value = values.get(start..end).ok_or_else(|| {
            ObservationError::Invalid("resolved field index is out of bounds".into())
        })?;
        selected.extend(project_component(value, component)?);
    }
    Ok(selected)
}

fn project_component(
    value: &[f64],
    component: ResolvedComponentProjection,
) -> Result<Vec<f64>, ObservationError> {
    match component {
        ResolvedComponentProjection::All => Ok(value.to_vec()),
        ResolvedComponentProjection::Cartesian { axis } => value
            .get(axis)
            .copied()
            .map(|value| vec![value])
            .ok_or(ObservationError::Identity(
                "resolved component exceeds field shape",
            )),
        ResolvedComponentProjection::Magnitude3 if value.len() == 3 => Ok(vec![value
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt()]),
        _ => Err(ObservationError::Unsupported(
            "component projection is not implemented for this recovered value shape",
        )),
    }
}

fn reduce_volume(
    observation: &ResolvedObservation,
    values: &[f64],
    components: usize,
    element_indices: &[usize],
    projection: &AnalysisProjection,
) -> Result<Vec<f64>, ObservationError> {
    let operation = reduction_operation(observation)?;
    ensure_integral_reduction(operation)?;
    let output_components = observation.result_component_count;
    let mut accumulator = vec![0.0; output_components];
    let mut measure = 0.0;
    for &element_index in element_indices {
        let element = projection.elements.get(element_index).ok_or_else(|| {
            ObservationError::Invalid("resolved element index is out of bounds".into())
        })?;
        match observation.spec.source_location {
            FieldLocation::Node => {
                let mut component_error = None;
                projection.visit_element_quadrature(element, |point| {
                    match interpolate_and_project(
                        values,
                        &element.node_indices,
                        point.shape_values(),
                        components,
                        observation.component,
                    ) {
                        Ok(projected) => accumulate(
                            &mut accumulator,
                            &projected,
                            point.volume_weight_m3,
                            operation,
                        ),
                        Err(error) => component_error = Some(error),
                    }
                    measure += point.volume_weight_m3;
                })?;
                if let Some(error) = component_error {
                    return Err(error);
                }
            }
            FieldLocation::Element => {
                let start = element_index
                    .checked_mul(components)
                    .ok_or(ObservationError::Limit("element field offset"))?;
                let end = start
                    .checked_add(components)
                    .ok_or(ObservationError::Limit("element field offset"))?;
                let projected = project_component(
                    values.get(start..end).ok_or_else(|| {
                        ObservationError::Invalid("element field index is out of bounds".into())
                    })?,
                    observation.component,
                )?;
                accumulate(
                    &mut accumulator,
                    &projected,
                    element.signed_volume_m3,
                    operation,
                );
                measure += element.signed_volume_m3;
            }
            _ => return Err(ObservationError::Unsupported("unsupported field location")),
        }
    }
    finish_reduction(accumulator, measure, operation)
}

fn reduce_boundary(
    observation: &ResolvedObservation,
    values: &[f64],
    components: usize,
    face_indices: &[usize],
    projection: &AnalysisProjection,
) -> Result<Vec<f64>, ObservationError> {
    let operation = reduction_operation(observation)?;
    ensure_integral_reduction(operation)?;
    let mut accumulator = vec![0.0; observation.result_component_count];
    let mut measure = 0.0;
    for &face_index in face_indices {
        let face = projection.boundary_faces.get(face_index).ok_or_else(|| {
            ObservationError::Invalid("resolved boundary-face index is out of bounds".into())
        })?;
        let mut component_error = None;
        projection.visit_boundary_quadrature(face, |point| {
            match interpolate_and_project(
                values,
                &face.node_indices,
                point.shape_values(),
                components,
                observation.component,
            ) {
                Ok(projected) => accumulate(
                    &mut accumulator,
                    &projected,
                    point.area_weight_m2,
                    operation,
                ),
                Err(error) => component_error = Some(error),
            }
            measure += point.area_weight_m2;
        })?;
        if let Some(error) = component_error {
            return Err(error);
        }
    }
    finish_reduction(accumulator, measure, operation)
}

fn interpolate_and_project(
    values: &[f64],
    entity_nodes: &[usize],
    shape_values: &[f64],
    components: usize,
    projection: ResolvedComponentProjection,
) -> Result<Vec<f64>, ObservationError> {
    let mut interpolated = vec![0.0; components];
    for (&node, shape) in entity_nodes.iter().zip(shape_values) {
        let start = node
            .checked_mul(components)
            .ok_or(ObservationError::Limit("nodal field offset"))?;
        let end = start
            .checked_add(components)
            .ok_or(ObservationError::Limit("nodal field offset"))?;
        let nodal = values.get(start..end).ok_or_else(|| {
            ObservationError::Invalid("nodal field index is out of bounds".into())
        })?;
        for (interpolated, nodal) in interpolated.iter_mut().zip(nodal) {
            *interpolated += nodal * shape;
        }
    }
    project_component(&interpolated, projection)
}

fn reduction_operation(
    observation: &ResolvedObservation,
) -> Result<SpatialReduction, ObservationError> {
    observation
        .spec
        .reduction
        .as_ref()
        .map(|reduction| reduction.spatial)
        .ok_or_else(|| ObservationError::Invalid("resolved reduction has no operation".into()))
}

fn ensure_integral_reduction(operation: SpatialReduction) -> Result<(), ObservationError> {
    if matches!(
        operation,
        SpatialReduction::Integral | SpatialReduction::Average | SpatialReduction::RootMeanSquare
    ) {
        Ok(())
    } else {
        Err(ObservationError::Unsupported(
            "this slice supports FE-weighted integral, average, and root-mean-square reductions",
        ))
    }
}

fn accumulate(accumulator: &mut [f64], value: &[f64], weight: f64, operation: SpatialReduction) {
    for (sum, value) in accumulator.iter_mut().zip(value) {
        *sum += if operation == SpatialReduction::RootMeanSquare {
            value * value * weight
        } else {
            value * weight
        };
    }
}

fn finish_reduction(
    mut accumulator: Vec<f64>,
    measure: f64,
    operation: SpatialReduction,
) -> Result<Vec<f64>, ObservationError> {
    if !measure.is_finite() || measure <= 0.0 {
        return Err(ObservationError::Invalid(
            "reduction target has no positive finite measure".into(),
        ));
    }
    match operation {
        SpatialReduction::Integral => {}
        SpatialReduction::Average => {
            accumulator.iter_mut().for_each(|value| *value /= measure);
        }
        SpatialReduction::RootMeanSquare => {
            accumulator
                .iter_mut()
                .for_each(|value| *value = (*value / measure).sqrt());
        }
        _ => {
            return Err(ObservationError::Unsupported(
                "unsupported spatial reduction",
            ))
        }
    }
    Ok(accumulator)
}
