use std::collections::BTreeSet;

use runmat_analysis_core::{ComplexRepresentation, FieldLocation, SpatialReduction};

use crate::{results::field::RecoveredFieldPartitionSet, topology::AnalysisProjection};

use super::{
    contract::observation_result_storage_component_count, topology::StructuralObservationTopology,
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
        observation.result_component_count,
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
    let mut accumulator = WeightedAccumulator::new(observation_result_storage_component_count(
        observation.result_component_count,
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
                        Ok(projected) => accumulator.add(&projected, point.volume_weight_m3),
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
                accumulator.add(&projected, element.signed_volume_m3);
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
    let mut accumulator = WeightedAccumulator::new(observation_result_storage_component_count(
        observation.result_component_count,
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
                Ok(projected) => accumulator.add(&projected, point.area_weight_m2),
                Err(error) => component_error = Some(error),
            }
        })?;
        if let Some(error) = component_error {
            return Err(error);
        }
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

    fn add(&mut self, values: &[f64], weight: f64) {
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
