//! Volume-weighted stress-recovery error estimation for canonical solid meshes.

use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, AnalysisFieldValues};
use runmat_meshing_core::{
    ElementOrder, FieldTopologyLocation, PersistentEntityId, SolverMeshArtifact, StableDigest,
};

use crate::assembly::elements::solid::{
    tetrahedron10::{extrapolate_quadrature_to_nodes, interpolate_nodal_samples},
    Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
};
use crate::contracts::{
    FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, FEA_FIELD_STRUCTURAL_STRESS,
};
use crate::progress::is_cancelled;
use crate::topology::quadrature::TETRAHEDRON_FOUR_POINT_RULE as QUADRATURE;

mod contract;
mod element;
mod element_norm;

pub use contract::{
    StructuralRecoveryEstimate, StructuralRecoveryEstimatorError,
    StructuralRecoveryEstimatorOptions, StructuralRecoveryIndicator, StructuralRecoveryNorm,
    StructuralRecoveryStatistics,
};

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

pub fn estimate_structural_recovery_error(
    artifact: &SolverMeshArtifact,
    stress_topology_id: &str,
    stress: &AnalysisField,
    options: StructuralRecoveryEstimatorOptions,
) -> Result<StructuralRecoveryEstimate, StructuralRecoveryEstimatorError> {
    validate_options(options)?;
    if is_cancelled() {
        return Err(StructuralRecoveryEstimatorError::Cancelled);
    }
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
    match validate_stress(stress, topology.ordered_entity_ids.len())? {
        ValidatedStress::Element(values) => {
            element::estimate(artifact, stress_topology_id, topology, values, options)
        }
        ValidatedStress::IntegrationPoint(values) => estimate_integration_point_recovery_error(
            artifact,
            stress_topology_id,
            topology,
            values,
            options,
        ),
    }
}

/// Physical stress is [xx, yy, zz, yz, xz, xy], with tensor shear components.
/// The symmetric matrix contains each shear component twice.
fn stress_norm_squared(values: [f64; STRESS_COMPONENT_COUNT]) -> f64 {
    values
        .into_iter()
        .enumerate()
        .map(|(component, value)| (if component < 3 { 1.0 } else { 2.0 }) * value * value)
        .sum()
}

fn finish_estimate(
    solver_artifact_digest: StableDigest,
    stress_topology_id: &str,
    mut indicators: Vec<StructuralRecoveryIndicator>,
    reference_norm: f64,
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
        norm: StructuralRecoveryNorm::StressTensorFrobeniusL2,
        solver_artifact_digest,
        stress_topology_id: stress_topology_id.to_owned(),
        total_error: total_squared.sqrt(),
        reference_norm,
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
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let mut region_side_nodal_stress =
        BTreeMap::<(u64, &PersistentEntityId), ([f64; STRESS_COMPONENT_COUNT], f64)>::new();
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
                .entry((node_id, &element.region_id))
                .or_insert(([0.0; STRESS_COMPONENT_COUNT], 0.0));
            for (sum, value) in sum.iter_mut().zip(value) {
                *sum += volume * value;
            }
            *weight += volume;
        }
    }

    let mut reference_squared_total = 0.0;
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
                    .get(&(*node_id, &element.region_id))
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
                        stress_norm_squared(std::array::from_fn(|component| {
                            sample[component] - recovered[component]
                        })) * volume
                            / INTEGRATION_POINT_COUNT as f64
                    })
                    .sum::<f64>();
                let reference = samples
                    .iter()
                    .map(|sample| stress_norm_squared(*sample))
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
                        stress_norm_squared(std::array::from_fn(|component| {
                            sample[component] - recovered[component]
                        })) * weight
                    })
                    .sum::<f64>();
                let reference = integration
                    .iter()
                    .zip(samples)
                    .map(|((_, weight), sample)| stress_norm_squared(*sample) * weight)
                    .sum::<f64>();
                (error, reference)
            }
        };
        // Overflow cannot become zero error through an infinite roundoff floor.
        if !error_squared.is_finite() || !reference_squared.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
        reference_squared_total += reference_squared;
        if !reference_squared_total.is_finite() {
            return Err(StructuralRecoveryEstimatorError::InvalidStressField);
        }
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
        reference_squared_total.sqrt(),
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
pub(crate) mod qualification_tests;
#[cfg(test)]
mod tests;
