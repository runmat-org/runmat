//! Volume-weighted stress-recovery error estimation for canonical solid meshes.

use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{ElementOrder, FieldTopologyLocation, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

use crate::assembly::elements::solid::{
    tetrahedron10::{extrapolate_quadrature_to_nodes, interpolate_nodal_samples},
    Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
};
use crate::contracts::{
    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, FEA_FIELD_STRUCTURAL_STRESS,
};
use crate::progress::is_cancelled;
use crate::topology::quadrature::TETRAHEDRON_FOUR_POINT_RULE as QUADRATURE;

const STRESS_COMPONENT_COUNT: usize = 6;
const INTEGRATION_POINT_COUNT: usize = 4;
const RECOVERY_ROUNDOFF_RELATIVE_TOLERANCE: f64 = 128.0 * f64::EPSILON;

enum ValidatedStress<'a> {
    Element(&'a [f64]),
    IntegrationPoint(&'a [f64]),
}

enum StressLayout {
    Element,
    IntegrationPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryEstimatorOptions {
    pub marking_fraction: f64,
    pub maximum_marked_elements: u64,
    pub cancellation_check_interval: u64,
}

impl Default for StructuralRecoveryEstimatorOptions {
    fn default() -> Self {
        Self {
            marking_fraction: 0.5,
            maximum_marked_elements: 1_000_000_000,
            cancellation_check_interval: 1_024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryIndicator {
    pub element_stable_identity: StableDigest,
    pub error: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryStatistics {
    pub element_count: u64,
    pub minimum_error: f64,
    pub maximum_error: f64,
    pub mean_error: f64,
    pub root_mean_square_error: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRecoveryEstimate {
    pub solver_artifact_digest: StableDigest,
    pub stress_topology_id: String,
    pub total_error: f64,
    pub indicators: Vec<StructuralRecoveryIndicator>,
    pub marked_element_identities: Vec<StableDigest>,
    pub statistics: StructuralRecoveryStatistics,
}

impl StructuralRecoveryEstimate {
    pub fn validate(&self) -> Result<(), StructuralRecoveryEstimatorError> {
        if self.solver_artifact_digest == StableDigest::ZERO
            || self.stress_topology_id.is_empty()
            || self.stress_topology_id.len() > 256
            || !self.stress_topology_id.is_ascii()
            || self.stress_topology_id.chars().any(char::is_control)
            || self.indicators.is_empty()
            || self.statistics.element_count != self.indicators.len() as u64
            || !self.total_error.is_finite()
            || self.total_error < 0.0
            || [
                self.statistics.minimum_error,
                self.statistics.maximum_error,
                self.statistics.mean_error,
                self.statistics.root_mean_square_error,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
        }
        let mut previous = None;
        let mut sum = 0.0;
        let mut sum_squared = 0.0;
        for indicator in &self.indicators {
            if indicator.element_stable_identity == StableDigest::ZERO
                || !indicator.error.is_finite()
                || indicator.error < 0.0
                || previous.is_some_and(|identity| identity >= indicator.element_stable_identity)
            {
                return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
            }
            sum += indicator.error;
            sum_squared += indicator.error * indicator.error;
            previous = Some(indicator.element_stable_identity);
        }
        let count = self.indicators.len() as f64;
        let expected = StructuralRecoveryStatistics {
            element_count: self.indicators.len() as u64,
            minimum_error: self
                .indicators
                .iter()
                .map(|indicator| indicator.error)
                .reduce(f64::min)
                .unwrap_or(0.0),
            maximum_error: self
                .indicators
                .iter()
                .map(|indicator| indicator.error)
                .reduce(f64::max)
                .unwrap_or(0.0),
            mean_error: sum / count,
            root_mean_square_error: (sum_squared / count).sqrt(),
        };
        let admitted = self
            .indicators
            .iter()
            .filter(|indicator| indicator.error > 0.0)
            .map(|indicator| indicator.element_stable_identity)
            .collect::<std::collections::BTreeSet<_>>();
        if self.total_error != sum_squared.sqrt()
            || self.statistics != expected
            || (self.total_error == 0.0) != self.marked_element_identities.is_empty()
            || !self
                .marked_element_identities
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self
                .marked_element_identities
                .iter()
                .any(|identity| !admitted.contains(identity))
        {
            return Err(StructuralRecoveryEstimatorError::InvalidEstimate);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralRecoveryEstimatorError {
    InvalidOptions,
    InvalidArtifact(String),
    MissingElementTopology,
    InvalidStressField,
    DeviceFieldRequiresHostTransfer,
    InvalidElementGeometry,
    InvalidEstimate,
    ResourceLimit,
    Cancelled,
}

pub fn estimate_structural_recovery_error(
    artifact: &SolverMeshArtifact,
    stress_topology_id: &str,
    stress: &AnalysisField,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<StructuralRecoveryEstimate, StructuralRecoveryEstimatorError> {
    validate_options(options)?;
    artifact.validate().map_err(|failure| {
        StructuralRecoveryEstimatorError::InvalidArtifact(failure.to_string())
    })?;
    let topology = artifact
        .topology
        .field_topologies
        .iter()
        .find(|topology| topology.topology_id == stress_topology_id)
        .filter(|topology| topology.location == FieldTopologyLocation::VolumeElement)
        .ok_or(StructuralRecoveryEstimatorError::MissingElementTopology)?;
    let values = match validate_stress(stress, topology.ordered_entity_ids.len())? {
        ValidatedStress::Element(values) => values,
        ValidatedStress::IntegrationPoint(values) => {
            return estimate_integration_point_recovery_error(
                artifact,
                stress_topology_id,
                topology,
                values,
                options,
            )
        }
    };
    let stress_by_element = topology
        .ordered_entity_ids
        .iter()
        .enumerate()
        .map(|(index, element_id)| {
            let offset = index * STRESS_COMPONENT_COUNT;
            (
                *element_id,
                &values[offset..offset + STRESS_COMPONENT_COUNT],
            )
        })
        .collect::<BTreeMap<_, _>>();
    let coordinates = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut region_side_nodal_stress =
        BTreeMap::<(u64, String), ([f64; STRESS_COMPONENT_COUNT], f64)>::new();
    let mut element_volumes = BTreeMap::new();
    let mut work = 0_u64;
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let volume = tetrahedron_volume(element, &coordinates)?;
        element_volumes.insert(element.element_id, volume);
        let element_stress = stress_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?;
        for node_id in &element.node_ids {
            let (sum, weight) = region_side_nodal_stress
                .entry((*node_id, element.region_id.source_topology_id.clone()))
                .or_insert(([0.0; STRESS_COMPONENT_COUNT], 0.0));
            for (sum, value) in sum.iter_mut().zip(*element_stress) {
                *sum += volume * value;
            }
            *weight += volume;
        }
    }

    let mut indicators = Vec::with_capacity(artifact.topology.volume_elements.len());
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let mut recovered = [0.0; STRESS_COMPONENT_COUNT];
        for node_id in &element.node_ids {
            let (sum, weight) = region_side_nodal_stress
                .get(&(*node_id, element.region_id.source_topology_id.clone()))
                .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
            if !weight.is_finite() || *weight <= 0.0 {
                return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
            }
            for (recovered, sum) in recovered.iter_mut().zip(sum) {
                *recovered += sum / weight / element.node_ids.len() as f64;
            }
        }
        let error_squared = stress_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?
            .iter()
            .zip(recovered)
            .map(|(value, recovered)| (value - recovered).powi(2))
            .sum::<f64>()
            * element_volumes
                .get(&element.element_id)
                .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
        let error = error_squared.sqrt();
        if !error.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
        indicators.push(StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error,
        });
    }
    finish_estimate(
        artifact.canonical_digest,
        stress_topology_id,
        indicators,
        options,
    )
}

fn finish_estimate(
    solver_artifact_digest: StableDigest,
    stress_topology_id: &str,
    mut indicators: Vec<StructuralRecoveryIndicator>,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<StructuralRecoveryEstimate, StructuralRecoveryEstimatorError> {
    indicators.sort_by_key(|indicator| indicator.element_stable_identity);
    let total_squared = indicators
        .iter()
        .map(|indicator| indicator.error * indicator.error)
        .sum::<f64>();
    let marked_element_identities = mark_elements(&indicators, total_squared, options)?;
    let count = indicators.len() as f64;
    let sum = indicators
        .iter()
        .map(|indicator| indicator.error)
        .sum::<f64>();
    let statistics = StructuralRecoveryStatistics {
        element_count: indicators.len() as u64,
        minimum_error: indicators
            .iter()
            .map(|indicator| indicator.error)
            .reduce(f64::min)
            .unwrap_or(0.0),
        maximum_error: indicators
            .iter()
            .map(|indicator| indicator.error)
            .reduce(f64::max)
            .unwrap_or(0.0),
        mean_error: sum / count,
        root_mean_square_error: (total_squared / count).sqrt(),
    };
    let estimate = StructuralRecoveryEstimate {
        solver_artifact_digest,
        stress_topology_id: stress_topology_id.to_owned(),
        total_error: total_squared.sqrt(),
        indicators,
        marked_element_identities,
        statistics,
    };
    estimate.validate()?;
    Ok(estimate)
}

fn estimate_integration_point_recovery_error(
    artifact: &SolverMeshArtifact,
    stress_topology_id: &str,
    topology: &runmat_meshing_core::FieldTopologyMap,
    values: &[f64],
    options: StructuralRecoveryEstimatorOptions,
) -> Result<StructuralRecoveryEstimate, StructuralRecoveryEstimatorError> {
    let samples_by_element = topology
        .ordered_entity_ids
        .iter()
        .enumerate()
        .map(|(index, element_id)| {
            let offset = index * INTEGRATION_POINT_COUNT * STRESS_COMPONENT_COUNT;
            let samples = std::array::from_fn(|point| {
                std::array::from_fn(|component| {
                    values[offset + point * STRESS_COMPONENT_COUNT + component]
                })
            });
            (*element_id, samples)
        })
        .collect::<BTreeMap<_, _>>();
    let coordinates = artifact
        .topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut region_side_nodal_stress =
        BTreeMap::<(u64, String), ([f64; STRESS_COMPONENT_COUNT], f64)>::new();
    let mut work = 0_u64;
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let samples = samples_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?;
        let volume = tetrahedron_volume(element, &coordinates)?;
        let nodal = match element.order {
            ElementOrder::Tet4 => {
                let average = std::array::from_fn(|component| {
                    samples.iter().map(|sample| sample[component]).sum::<f64>()
                        / INTEGRATION_POINT_COUNT as f64
                });
                vec![average; 4]
            }
            ElementOrder::Tet10 => extrapolate_quadrature_to_nodes(*samples).to_vec(),
        };
        if nodal.len() != element.node_ids.len() {
            return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
        }
        for (&node_id, value) in element.node_ids.iter().zip(nodal) {
            let (sum, weight) = region_side_nodal_stress
                .entry((node_id, element.region_id.source_topology_id.clone()))
                .or_insert(([0.0; STRESS_COMPONENT_COUNT], 0.0));
            for (sum, value) in sum.iter_mut().zip(value) {
                *sum += volume * value;
            }
            *weight += volume;
        }
    }

    let mut indicators = Vec::with_capacity(artifact.topology.volume_elements.len());
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let samples = samples_by_element
            .get(&element.element_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidStressField)?;
        let nodal = element
            .node_ids
            .iter()
            .map(|node_id| {
                let (sum, weight) = region_side_nodal_stress
                    .get(&(*node_id, element.region_id.source_topology_id.clone()))
                    .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
                if !weight.is_finite() || *weight <= 0.0 {
                    return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
                }
                Ok(sum.map(|value| value / weight))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (error_squared, reference_squared) = match element.order {
            ElementOrder::Tet4 => {
                if nodal.len() != 4 {
                    return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
                }
                let volume = tetrahedron_volume(element, &coordinates)?;
                let error = QUADRATURE
                    .iter()
                    .zip(samples)
                    .map(|(barycentric, sample)| {
                        let recovered =
                            std::array::from_fn::<_, STRESS_COMPONENT_COUNT, _>(|component| {
                                (0..4)
                                    .map(|node| barycentric[node] * nodal[node][component])
                                    .sum::<f64>()
                            });
                        sample
                            .iter()
                            .zip(recovered)
                            .map(|(value, recovered)| (value - recovered).powi(2))
                            .sum::<f64>()
                            * volume
                            / INTEGRATION_POINT_COUNT as f64
                    })
                    .sum::<f64>();
                let reference = samples
                    .iter()
                    .flat_map(|sample| sample.iter())
                    .map(|value| value * value)
                    .sum::<f64>()
                    * volume
                    / INTEGRATION_POINT_COUNT as f64;
                (error, reference)
            }
            ElementOrder::Tet10 => {
                let geometry = Tetrahedron10ElementGeometry {
                    nodes_m: element_coordinates::<10>(element, &coordinates)?,
                };
                let nodal: [[f64; STRESS_COMPONENT_COUNT]; 10] = nodal
                    .try_into()
                    .map_err(|_| StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
                let integration = geometry
                    .quadrature_integration_data()
                    .map_err(|_| StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
                let error = integration
                    .iter()
                    .zip(samples)
                    .map(|((barycentric, weight), sample)| {
                        let recovered = interpolate_nodal_samples(nodal, *barycentric);
                        sample
                            .iter()
                            .zip(recovered)
                            .map(|(value, recovered)| (value - recovered).powi(2))
                            .sum::<f64>()
                            * weight
                    })
                    .sum::<f64>();
                let reference = integration
                    .iter()
                    .zip(samples)
                    .map(|((_, weight), sample)| {
                        sample.iter().map(|value| value * value).sum::<f64>() * weight
                    })
                    .sum::<f64>();
                (error, reference)
            }
        };
        let noise_floor = RECOVERY_ROUNDOFF_RELATIVE_TOLERANCE.powi(2) * reference_squared;
        let error = if error_squared <= noise_floor {
            0.0
        } else {
            error_squared.sqrt()
        };
        if !error.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
        indicators.push(StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error,
        });
    }
    finish_estimate(
        artifact.canonical_digest,
        stress_topology_id,
        indicators,
        options,
    )
}

fn validate_options(
    options: StructuralRecoveryEstimatorOptions,
) -> Result<(), StructuralRecoveryEstimatorError> {
    if !options.marking_fraction.is_finite()
        || !(0.0..=1.0).contains(&options.marking_fraction)
        || options.marking_fraction == 0.0
        || options.maximum_marked_elements == 0
        || options.cancellation_check_interval == 0
    {
        return Err(StructuralRecoveryEstimatorError::InvalidOptions);
    }
    Ok(())
}

fn validate_stress(
    stress: &AnalysisField,
    element_count: usize,
) -> Result<ValidatedStress<'_>, StructuralRecoveryEstimatorError> {
    let kind = if stress.field_id == FEA_FIELD_STRUCTURAL_STRESS
        && stress.shape == [element_count, STRESS_COMPONENT_COUNT]
    {
        StressLayout::Element
    } else if stress.field_id == FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
        && stress.shape
            == [
                element_count * INTEGRATION_POINT_COUNT,
                STRESS_COMPONENT_COUNT,
            ]
    {
        StressLayout::IntegrationPoint
    } else {
        return Err(StructuralRecoveryEstimatorError::InvalidStressField);
    };
    match &stress.values {
        AnalysisFieldValues::HostF64(values)
            if values.len() == stress.shape.iter().product::<usize>()
                && values.iter().all(|value| value.is_finite()) =>
        {
            Ok(match kind {
                StressLayout::Element => ValidatedStress::Element(values),
                StressLayout::IntegrationPoint => ValidatedStress::IntegrationPoint(values),
            })
        }
        AnalysisFieldValues::HostF64(_) => {
            Err(StructuralRecoveryEstimatorError::InvalidStressField)
        }
        AnalysisFieldValues::DeviceRef(_) => {
            Err(StructuralRecoveryEstimatorError::DeviceFieldRequiresHostTransfer)
        }
    }
}

fn tetrahedron_volume(
    element: &runmat_meshing_core::SolverVolumeElement,
    coordinates: &BTreeMap<u64, [f64; 3]>,
) -> Result<f64, StructuralRecoveryEstimatorError> {
    match element.order {
        ElementOrder::Tet4 => Tetrahedron4ElementGeometry {
            nodes_m: element_coordinates::<4>(element, coordinates)?,
        }
        .volume_m3()
        .map_err(|_| StructuralRecoveryEstimatorError::InvalidElementGeometry),
        ElementOrder::Tet10 => Tetrahedron10ElementGeometry {
            nodes_m: element_coordinates::<10>(element, coordinates)?,
        }
        .volume_m3()
        .map_err(|_| StructuralRecoveryEstimatorError::InvalidElementGeometry),
    }
}

fn element_coordinates<const N: usize>(
    element: &runmat_meshing_core::SolverVolumeElement,
    coordinates: &BTreeMap<u64, [f64; 3]>,
) -> Result<[[f64; 3]; N], StructuralRecoveryEstimatorError> {
    if element.node_ids.len() != N {
        return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
    }
    let mut nodes_m = [[0.0; 3]; N];
    for (local, node_id) in element.node_ids.iter().enumerate() {
        nodes_m[local] = *coordinates
            .get(node_id)
            .ok_or(StructuralRecoveryEstimatorError::InvalidElementGeometry)?;
    }
    Ok(nodes_m)
}

fn mark_elements(
    indicators: &[StructuralRecoveryIndicator],
    total_squared: f64,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<Vec<StableDigest>, StructuralRecoveryEstimatorError> {
    if total_squared == 0.0 {
        return Ok(Vec::new());
    }
    let mut ranked = indicators.to_vec();
    ranked.sort_by(|left, right| {
        right.error.total_cmp(&left.error).then_with(|| {
            left.element_stable_identity
                .cmp(&right.element_stable_identity)
        })
    });
    let target = options.marking_fraction * total_squared;
    let mut admitted = 0.0;
    let mut marked = Vec::new();
    for indicator in ranked {
        if marked.len() as u64 >= options.maximum_marked_elements {
            return Err(StructuralRecoveryEstimatorError::ResourceLimit);
        }
        admitted += indicator.error * indicator.error;
        marked.push(indicator.element_stable_identity);
        if admitted >= target {
            break;
        }
    }
    marked.sort_unstable();
    Ok(marked)
}

fn checkpoint(
    work: &mut u64,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<(), StructuralRecoveryEstimatorError> {
    if work.is_multiple_of(options.cancellation_check_interval) && is_cancelled() {
        return Err(StructuralRecoveryEstimatorError::Cancelled);
    }
    *work = work.saturating_add(1);
    Ok(())
}

#[cfg(test)]
mod qualification_tests;
#[cfg(test)]
mod tests;
