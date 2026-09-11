use std::collections::BTreeSet;

use runmat_analysis_core::{
    ComplexRepresentation, FieldLocation, ObservationValueShape, SignConvention, SpatialReduction,
};

use crate::{results::field::RecoveredFieldPartitionSet, topology::AnalysisProjection};

use super::{
    contract::{observation_result_storage_component_count, shape_component_count},
    topology::StructuralObservationTopology,
    ObservationError, ObservationPartition, ObservationPartitionPayload,
    ObservationPartitionTarget, ResolvedComponentProjection, ResolvedObservation,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CompensatedSum {
    sum: f64,
    correction: f64,
}

pub(super) fn required_source_indices(
    observation: &ResolvedObservation,
    partition: &ObservationPartition,
    topology: &StructuralObservationTopology,
) -> Result<Vec<usize>, ObservationError> {
    let indices = match &partition.target {
        ObservationPartitionTarget::Nodes { indices, .. }
        | ObservationPartitionTarget::Elements { indices, .. } => indices.clone(),
        ObservationPartitionTarget::Points {
            sample_ordinals, ..
        } => {
            let super::ResolvedObservationTarget::Points { samples } = &observation.target else {
                return Err(ObservationError::Identity(
                    "point partition does not match its resolved target",
                ));
            };
            let mut nodes = BTreeSet::new();
            for ordinal in sample_ordinals {
                let sample = samples.get(*ordinal).ok_or(ObservationError::Identity(
                    "point partition exceeds its resolved target",
                ))?;
                nodes.extend(sample.source_node_indices.iter().copied());
            }
            nodes.into_iter().collect()
        }
        ObservationPartitionTarget::Volume {
            element_indices, ..
        } => match observation.spec.source_location {
            FieldLocation::Node => {
                let projection =
                    topology
                        .solid_projection
                        .as_ref()
                        .ok_or(ObservationError::Unsupported(
                            "volume observations require admitted solid topology",
                        ))?;
                let mut nodes = BTreeSet::new();
                for index in element_indices {
                    let element =
                        projection
                            .elements
                            .get(*index)
                            .ok_or(ObservationError::Identity(
                                "observation partition exceeds solid element topology",
                            ))?;
                    nodes.extend(element.node_indices.iter().copied());
                }
                nodes.into_iter().collect()
            }
            FieldLocation::Element => element_indices.clone(),
            _ => {
                return Err(ObservationError::Unsupported(
                    "volume observation field location is unsupported",
                ))
            }
        },
        ObservationPartitionTarget::Boundary { face_indices, .. } => {
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "boundary observations require admitted solid topology",
                    ))?;
            let mut nodes = BTreeSet::new();
            for index in face_indices {
                let face =
                    projection
                        .boundary_faces
                        .get(*index)
                        .ok_or(ObservationError::Identity(
                            "observation partition exceeds boundary-face topology",
                        ))?;
                nodes.extend(face.node_indices.iter().copied());
            }
            nodes.into_iter().collect()
        }
    };
    if indices.is_empty() {
        return Err(ObservationError::Invalid(
            "observation partition requires at least one source field entity".into(),
        ));
    }
    Ok(indices)
}

pub(super) fn interpolate_values(
    field: &RecoveredFieldPartitionSet,
    sample_ordinals: &[usize],
    observation: &ResolvedObservation,
) -> Result<(Vec<super::ObservationEntity>, Vec<f64>), ObservationError> {
    let super::ResolvedObservationTarget::Points { samples } = &observation.target else {
        return Err(ObservationError::Identity(
            "point partition does not match its resolved target",
        ));
    };
    let mut entities = Vec::with_capacity(sample_ordinals.len());
    let mut values = Vec::new();
    for ordinal in sample_ordinals {
        let sample = samples.get(*ordinal).ok_or(ObservationError::Identity(
            "point partition exceeds its resolved target",
        ))?;
        let first_index = *sample
            .source_node_indices
            .first()
            .ok_or_else(|| ObservationError::Invalid("point sample has no source nodes".into()))?;
        let width = field
            .value(first_index)
            .ok_or(ObservationError::Identity(
                "point interpolation is missing an owned source node",
            ))?
            .len();
        let mut sums = vec![CompensatedSum::default(); width];
        for (node_index, weight) in sample.source_node_indices.iter().zip(&sample.shape_values) {
            let nodal = field.value(*node_index).ok_or(ObservationError::Identity(
                "point interpolation is missing an owned source node",
            ))?;
            if nodal.len() != width {
                return Err(ObservationError::Identity(
                    "point interpolation source values have inconsistent shapes",
                ));
            }
            for (sum, component) in sums.iter_mut().zip(nodal) {
                sum.add(weight * component);
            }
        }
        let interpolated = sums
            .into_iter()
            .map(CompensatedSum::total)
            .collect::<Vec<_>>();
        values.extend(project_observation_value(
            field,
            &interpolated,
            observation,
        )?);
        entities.push(super::ObservationEntity::Point {
            identity: sample.identity,
            ordinal: sample.ordinal,
            position_m: sample.position_m,
            source_element: sample.source_element_identity,
        });
    }
    Ok((entities, values))
}

impl CompensatedSum {
    pub(super) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    pub(super) fn total(self) -> f64 {
        self.sum + self.correction
    }
}

pub(super) fn select_values(
    field: &RecoveredFieldPartitionSet,
    indices: &[usize],
    observation: &ResolvedObservation,
) -> Result<Vec<f64>, ObservationError> {
    let components = observation_result_storage_component_count(
        shape_component_count(&observation.result_value_shape)?,
        observation.spec.complex_representation,
    )?;
    let mut selected = Vec::with_capacity(indices.len().saturating_mul(components));
    for &index in indices {
        let value = field.value(index).ok_or(ObservationError::Identity(
            "observation source partition is missing an owned field entity",
        ))?;
        selected.extend(project_observation_value(field, value, observation)?);
    }
    Ok(selected)
}

fn project_component(
    value: &[f64],
    component: ResolvedComponentProjection,
    complex_output: bool,
) -> Result<Vec<f64>, ObservationError> {
    match component {
        ResolvedComponentProjection::All => Ok(value.to_vec()),
        ResolvedComponentProjection::Cartesian { axis } if complex_output => {
            let start = axis
                .checked_mul(2)
                .ok_or(ObservationError::Limit("complex Cartesian component index"))?;
            value
                .get(start..start + 2)
                .map(<[f64]>::to_vec)
                .ok_or(ObservationError::Identity(
                    "resolved component exceeds complex field shape",
                ))
        }
        ResolvedComponentProjection::Cartesian { axis } => value
            .get(axis)
            .copied()
            .map(|value| vec![value])
            .ok_or(ObservationError::Identity(
                "resolved component exceeds field shape",
            )),
        ResolvedComponentProjection::Magnitude3 if !complex_output && value.len() == 3 => {
            Ok(vec![value
                .iter()
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt()])
        }
        _ => Err(ObservationError::Unsupported(
            "component projection is not implemented for this recovered value shape",
        )),
    }
}

fn project_observation_value(
    field: &RecoveredFieldPartitionSet,
    value: &[f64],
    observation: &ResolvedObservation,
) -> Result<Vec<f64>, ObservationError> {
    use crate::results::field::RecoveredFieldNumericRepresentation;

    let represented = match field.definition.numeric_representation() {
        RecoveredFieldNumericRepresentation::Real
            if observation.spec.complex_representation == ComplexRepresentation::Real =>
        {
            value.to_vec()
        }
        RecoveredFieldNumericRepresentation::Real => {
            return Err(ObservationError::Invalid(
                "a real recovered field cannot provide imaginary, phase, or complex values".into(),
            ))
        }
        RecoveredFieldNumericRepresentation::ComplexInterleaved => {
            if !value.len().is_multiple_of(2) {
                return Err(ObservationError::Identity(
                    "complex recovered value does not contain interleaved pairs",
                ));
            }
            match observation.spec.complex_representation {
                ComplexRepresentation::Real => value.chunks_exact(2).map(|pair| pair[0]).collect(),
                ComplexRepresentation::Imaginary => {
                    value.chunks_exact(2).map(|pair| pair[1]).collect()
                }
                ComplexRepresentation::Magnitude => value
                    .chunks_exact(2)
                    .map(|pair| pair[0].hypot(pair[1]))
                    .collect(),
                ComplexRepresentation::PhaseRadians => value
                    .chunks_exact(2)
                    .map(|pair| pair[1].atan2(pair[0]))
                    .collect(),
                ComplexRepresentation::Complex => value.to_vec(),
            }
        }
    };
    project_component(
        &represented,
        observation.component,
        observation.spec.complex_representation == ComplexRepresentation::Complex,
    )
}

pub(super) fn weighted_volume_contribution(
    observation: &ResolvedObservation,
    field: &RecoveredFieldPartitionSet,
    element_indices: &[usize],
    projection: &AnalysisProjection,
) -> Result<ObservationPartitionPayload, ObservationError> {
    let operation = reduction_operation(observation)?;
    let mut accumulator = WeightedAccumulator::new(observation_result_storage_component_count(
        shape_component_count(&observation.result_value_shape)?,
        observation.spec.complex_representation,
    )?);
    for &element_index in element_indices {
        let element = projection.elements.get(element_index).ok_or_else(|| {
            ObservationError::Invalid("resolved element index is out of bounds".into())
        })?;
        match observation.spec.source_location {
            FieldLocation::Node => {
                let mut component_error = None;
                projection.visit_element_quadrature(
                    element,
                    |point| match interpolate_and_project(
                        field,
                        &element.node_indices,
                        point.shape_values(),
                        observation,
                    ) {
                        Ok(projected) => {
                            component_error = accumulate_volume_reduction(
                                &mut accumulator,
                                operation,
                                &projected,
                                point.position_m,
                                point.volume_weight_m3,
                            )
                            .err()
                        }
                        Err(error) => component_error = Some(error),
                    },
                )?;
                if let Some(error) = component_error {
                    return Err(error);
                }
            }
            FieldLocation::Element => {
                let value = field
                    .value(element_index)
                    .ok_or(ObservationError::Identity(
                        "observation source partition is missing an owned element field value",
                    ))?;
                let projected = project_observation_value(field, value, observation)?;
                accumulate_volume_reduction(
                    &mut accumulator,
                    operation,
                    &projected,
                    element.centroid_m,
                    element.signed_volume_m3,
                )?;
            }
            _ => return Err(ObservationError::Unsupported("unsupported field location")),
        }
    }
    accumulator.finish()
}

pub(super) fn weighted_boundary_contribution(
    observation: &ResolvedObservation,
    field: &RecoveredFieldPartitionSet,
    face_indices: &[usize],
    projection: &AnalysisProjection,
) -> Result<ObservationPartitionPayload, ObservationError> {
    let operation = reduction_operation(observation)?;
    let mut accumulator = WeightedAccumulator::new(observation_result_storage_component_count(
        shape_component_count(&observation.result_value_shape)?,
        observation.spec.complex_representation,
    )?);
    for &face_index in face_indices {
        let face = projection.boundary_faces.get(face_index).ok_or_else(|| {
            ObservationError::Invalid("resolved boundary-face index is out of bounds".into())
        })?;
        let mut component_error = None;
        projection.visit_boundary_quadrature(face, |point| {
            match interpolate_and_project(
                field,
                &face.node_indices,
                point.shape_values(),
                observation,
            ) {
                Ok(projected) => {
                    component_error = accumulate_boundary_reduction(
                        &mut accumulator,
                        observation,
                        operation,
                        &projected,
                        point.position_m,
                        point.area_vector_weight_m2,
                        point.area_weight_m2,
                    )
                    .err()
                }
                Err(error) => component_error = Some(error),
            }
        })?;
        if let Some(error) = component_error {
            return Err(error);
        }
    }
    accumulator.finish()
}

pub(super) fn discrete_reaction_contribution(
    observation: &ResolvedObservation,
    field: &RecoveredFieldPartitionSet,
    node_indices: &[usize],
    topology: &StructuralObservationTopology,
) -> Result<ObservationPartitionPayload, ObservationError> {
    if !matches!(
        observation.spec.quantity,
        runmat_analysis_core::ObservationQuantity::ReactionForce
            | runmat_analysis_core::ObservationQuantity::ReactionMoment
    ) {
        return Err(ObservationError::Invalid(
            "discrete resultant requires a reaction field".into(),
        ));
    }
    let operation = reduction_operation(observation)?;
    let mut accumulator = WeightedAccumulator::new(3);
    for &node_index in node_indices {
        let node = topology
            .nodes
            .get(node_index)
            .ok_or(ObservationError::Identity(
                "reaction reduction exceeds structural node topology",
            ))?;
        let value = field.value(node_index).ok_or(ObservationError::Identity(
            "reaction reduction is missing an owned nodal value",
        ))?;
        let projected = project_observation_value(field, value, observation)?;
        let vector = weighted_vector(&projected, 1.0)?;
        let contribution = match operation {
            SpatialReduction::Resultant => vector,
            SpatialReduction::Moment { about }
                if observation.spec.quantity
                    == runmat_analysis_core::ObservationQuantity::ReactionForce =>
            {
                cross(subtract(node.coordinates_m, about.meters), vector)
            }
            SpatialReduction::Moment { .. } => {
                return Err(ObservationError::Invalid(
                    "a moment reduction cannot take a second moment of reaction moments".into(),
                ))
            }
            _ => {
                return Err(ObservationError::Invalid(
                    "discrete reaction contribution requires a resultant or moment".into(),
                ))
            }
        };
        accumulator.add_resultant(contribution, 1.0)?;
    }
    accumulator.finish()
}

struct WeightedAccumulator {
    linear_sum: Vec<CompensatedSum>,
    square_sum: Vec<CompensatedSum>,
    measure: CompensatedSum,
}

impl WeightedAccumulator {
    fn new(component_count: usize) -> Self {
        Self {
            linear_sum: vec![CompensatedSum::default(); component_count],
            square_sum: vec![CompensatedSum::default(); component_count],
            measure: CompensatedSum::default(),
        }
    }

    fn add(&mut self, values: &[f64], weight: f64) -> Result<(), ObservationError> {
        if values.len() != self.linear_sum.len() || !weight.is_finite() || weight <= 0.0 {
            return Err(ObservationError::Invalid(
                "weighted reduction component count or measure is invalid".into(),
            ));
        }
        for ((linear, square), value) in self
            .linear_sum
            .iter_mut()
            .zip(&mut self.square_sum)
            .zip(values)
        {
            linear.add(value * weight);
            square.add(value * value * weight);
        }
        self.measure.add(weight);
        Ok(())
    }

    fn add_norm(&mut self, values: &[f64], weight: f64) -> Result<(), ObservationError> {
        if self.square_sum.len() != 1
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
            || !weight.is_finite()
            || weight <= 0.0
        {
            return Err(ObservationError::Invalid(
                "Euclidean-norm reduction input is invalid".into(),
            ));
        }
        self.square_sum[0].add(values.iter().map(|value| value * value).sum::<f64>() * weight);
        self.measure.add(weight);
        Ok(())
    }

    fn add_resultant(&mut self, value: [f64; 3], measure: f64) -> Result<(), ObservationError> {
        if self.linear_sum.len() != 3
            || value.iter().any(|component| !component.is_finite())
            || !measure.is_finite()
            || measure <= 0.0
        {
            return Err(ObservationError::Invalid(
                "resultant reduction input is invalid".into(),
            ));
        }
        for (sum, component) in self.linear_sum.iter_mut().zip(value) {
            sum.add(component);
        }
        self.measure.add(measure);
        Ok(())
    }

    fn finish(self) -> Result<ObservationPartitionPayload, ObservationError> {
        let measure = self.measure.total();
        let linear_sum = self
            .linear_sum
            .into_iter()
            .map(CompensatedSum::total)
            .collect::<Vec<_>>();
        let square_sum = self
            .square_sum
            .into_iter()
            .map(CompensatedSum::total)
            .collect::<Vec<_>>();
        if !measure.is_finite()
            || measure <= 0.0
            || linear_sum
                .iter()
                .chain(&square_sum)
                .any(|value| !value.is_finite())
        {
            return Err(ObservationError::Invalid(
                "partition reduction produced invalid weighted moments".into(),
            ));
        }
        Ok(ObservationPartitionPayload::WeightedReduction {
            linear_sum,
            square_sum,
            measure,
        })
    }
}

fn accumulate_volume_reduction(
    accumulator: &mut WeightedAccumulator,
    operation: SpatialReduction,
    projected: &[f64],
    position_m: [f64; 3],
    weight: f64,
) -> Result<(), ObservationError> {
    match operation {
        SpatialReduction::EuclideanNorm => accumulator.add_norm(projected, weight),
        SpatialReduction::Resultant => {
            accumulator.add_resultant(weighted_vector(projected, weight)?, weight)
        }
        SpatialReduction::Moment { about } => {
            let force = weighted_vector(projected, weight)?;
            accumulator.add_resultant(cross(subtract(position_m, about.meters), force), weight)
        }
        _ => accumulator.add(projected, weight),
    }
}

fn accumulate_boundary_reduction(
    accumulator: &mut WeightedAccumulator,
    observation: &ResolvedObservation,
    operation: SpatialReduction,
    projected: &[f64],
    position_m: [f64; 3],
    area_vector_weight_m2: [f64; 3],
    area_weight_m2: f64,
) -> Result<(), ObservationError> {
    match operation {
        SpatialReduction::EuclideanNorm => accumulator.add_norm(projected, area_weight_m2),
        SpatialReduction::Resultant | SpatialReduction::Moment { .. } => {
            let force = boundary_force(
                observation,
                projected,
                area_vector_weight_m2,
                area_weight_m2,
            )?;
            let value = match operation {
                SpatialReduction::Moment { about } => {
                    cross(subtract(position_m, about.meters), force)
                }
                SpatialReduction::Resultant => force,
                _ => unreachable!(),
            };
            accumulator.add_resultant(value, area_weight_m2)
        }
        _ => accumulator.add(projected, area_weight_m2),
    }
}

fn boundary_force(
    observation: &ResolvedObservation,
    projected: &[f64],
    area_vector_weight_m2: [f64; 3],
    area_weight_m2: f64,
) -> Result<[f64; 3], ObservationError> {
    match observation.spec.value_shape {
        ObservationValueShape::Vector3 => weighted_vector(projected, area_weight_m2),
        ObservationValueShape::SymmetricTensor3 if projected.len() == 6 => Ok([
            projected[0] * area_vector_weight_m2[0]
                + projected[3] * area_vector_weight_m2[1]
                + projected[5] * area_vector_weight_m2[2],
            projected[3] * area_vector_weight_m2[0]
                + projected[1] * area_vector_weight_m2[1]
                + projected[4] * area_vector_weight_m2[2],
            projected[5] * area_vector_weight_m2[0]
                + projected[4] * area_vector_weight_m2[1]
                + projected[2] * area_vector_weight_m2[2],
        ]),
        ObservationValueShape::Scalar if projected.len() == 1 => {
            let sign = match observation.spec.sign_convention {
                SignConvention::OutwardPositive => 1.0,
                SignConvention::InwardPositive => -1.0,
                _ => {
                    return Err(ObservationError::Invalid(
                        "pressure resultant lacks an explicit normal sign convention".into(),
                    ))
                }
            };
            Ok(area_vector_weight_m2.map(|component| sign * projected[0] * component))
        }
        _ => Err(ObservationError::Invalid(
            "resultant source shape differs from its resolved contract".into(),
        )),
    }
}

fn weighted_vector(values: &[f64], weight: f64) -> Result<[f64; 3], ObservationError> {
    let [x, y, z] = values else {
        return Err(ObservationError::Invalid(
            "vector resultant requires exactly three real components".into(),
        ));
    };
    Ok([x * weight, y * weight, z * weight])
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn interpolate_and_project(
    field: &RecoveredFieldPartitionSet,
    entity_nodes: &[usize],
    shape_values: &[f64],
    observation: &ResolvedObservation,
) -> Result<Vec<f64>, ObservationError> {
    let components = field
        .definition
        .storage_component_count()
        .map_err(|error| ObservationError::Invalid(error.to_string()))?;
    let mut interpolated = vec![0.0; components];
    for (&node, shape) in entity_nodes.iter().zip(shape_values) {
        let nodal = field.value(node).ok_or(ObservationError::Identity(
            "observation source partition is missing a required nodal value",
        ))?;
        for (interpolated, nodal) in interpolated.iter_mut().zip(nodal) {
            *interpolated += nodal * shape;
        }
    }
    project_observation_value(field, &interpolated, observation)
}

pub(super) fn reduction_operation(
    observation: &ResolvedObservation,
) -> Result<SpatialReduction, ObservationError> {
    observation
        .spec
        .reduction
        .as_ref()
        .map(|reduction| reduction.spatial)
        .ok_or_else(|| ObservationError::Invalid("resolved reduction has no operation".into()))
}
