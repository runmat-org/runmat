//! Volume-weighted stress-recovery error estimation for canonical solid meshes.

use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{FieldTopologyLocation, SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

use crate::contracts::FEA_FIELD_STRUCTURAL_STRESS;
use crate::progress::is_cancelled;

const STRESS_COMPONENT_COUNT: usize = 6;

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
    let values = validate_stress(stress, topology.ordered_entity_ids.len())?;
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
    let mut nodal_stress = BTreeMap::<u64, ([f64; STRESS_COMPONENT_COUNT], f64)>::new();
    let mut element_volumes = BTreeMap::new();
    let mut work = 0_u64;
    for element in &artifact.topology.volume_elements {
        checkpoint(&mut work, options)?;
        let volume = tetrahedron_volume(element, &coordinates)?;
        element_volumes.insert(element.element_id, volume);
        let element_stress = stress_by_element[&element.element_id];
        for node_id in &element.node_ids {
            let (sum, weight) = nodal_stress
                .entry(*node_id)
                .or_insert(([0.0; STRESS_COMPONENT_COUNT], 0.0));
            for (sum, value) in sum.iter_mut().zip(element_stress) {
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
            let (sum, weight) = &nodal_stress[node_id];
            for (recovered, sum) in recovered.iter_mut().zip(sum) {
                *recovered += sum / weight / element.node_ids.len() as f64;
            }
        }
        let error_squared = stress_by_element[&element.element_id]
            .iter()
            .zip(recovered)
            .map(|(value, recovered)| (value - recovered).powi(2))
            .sum::<f64>()
            * element_volumes[&element.element_id];
        let error = error_squared.sqrt();
        if !error.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
        indicators.push(StructuralRecoveryIndicator {
            element_stable_identity: element.stable_identity,
            error,
        });
    }
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
        solver_artifact_digest: artifact.canonical_digest,
        stress_topology_id: stress_topology_id.to_owned(),
        total_error: total_squared.sqrt(),
        indicators,
        marked_element_identities,
        statistics,
    };
    estimate.validate()?;
    Ok(estimate)
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
) -> Result<&[f64], StructuralRecoveryEstimatorError> {
    if stress.field_id != FEA_FIELD_STRUCTURAL_STRESS
        || stress.shape != [element_count, STRESS_COMPONENT_COUNT]
    {
        return Err(StructuralRecoveryEstimatorError::InvalidStressField);
    }
    match &stress.values {
        AnalysisFieldValues::HostF64(values)
            if values.len() == element_count * STRESS_COMPONENT_COUNT
                && values.iter().all(|value| value.is_finite()) =>
        {
            Ok(values)
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
    let [a, b, c, d] = std::array::from_fn(|index| coordinates[&element.node_ids[index]]);
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let ad = subtract(d, a);
    let determinant = dot(ab, cross(ac, ad));
    if !determinant.is_finite() || determinant <= 0.0 {
        return Err(StructuralRecoveryEstimatorError::InvalidElementGeometry);
    }
    Ok(determinant / 6.0)
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

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests;
