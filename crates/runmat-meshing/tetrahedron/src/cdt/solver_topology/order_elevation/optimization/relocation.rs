use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    SolverMeshTopology,
};

use super::{consume_candidate_work, jacobian, OptimizationWork};
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
};

const RELOCATION_FRACTIONS: [f64; 6] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75];

pub(in crate::cdt::solver_topology::order_elevation) fn relocate_interior_corner(
    topology: &mut SolverMeshTopology,
    element_index: usize,
    maximum_search_work: u64,
    maximum_recursion_depth: u32,
    work: OptimizationWork<'_>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let OptimizationWork {
        options,
        cancellation,
        jacobian_work,
        candidate_work,
    } = work;
    let element = topology
        .volume_elements
        .get(element_index)
        .ok_or_else(|| invalid("interior-corner repair element index is out of bounds"))?
        .clone();
    let boundary_nodes = topology
        .boundary_faces
        .iter()
        .flat_map(|face| face.node_ids.iter().take(3).copied())
        .collect::<BTreeSet<_>>();
    let node_index = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();

    for &node_id in element.node_ids.iter().take(4) {
        if boundary_nodes.contains(&node_id) {
            continue;
        }
        let index = *node_index
            .get(&node_id)
            .ok_or_else(|| invalid("interior-corner repair node is absent"))?;
        let incident_elements = topology
            .volume_elements
            .iter()
            .enumerate()
            .filter_map(|(element_index, candidate)| {
                candidate.node_ids[..4]
                    .contains(&node_id)
                    .then_some(element_index)
            })
            .collect::<BTreeSet<_>>();
        let original_signs = corner_signs(topology, &node_index, &incident_elements)?;
        let neighbor_ids = incident_elements
            .iter()
            .flat_map(|element_index| topology.volume_elements[*element_index].node_ids[..4].iter())
            .copied()
            .filter(|candidate| *candidate != node_id)
            .collect::<BTreeSet<_>>();
        if neighbor_ids.is_empty() {
            continue;
        }
        let target = neighbor_ids
            .iter()
            .try_fold([0.0; 3], |mut sum, neighbor_id| {
                let point = topology.nodes[*node_index
                    .get(neighbor_id)
                    .ok_or_else(|| invalid("interior-corner neighbor is absent"))?]
                .coordinates_m;
                for axis in 0..3 {
                    sum[axis] += point[axis];
                }
                Ok::<_, DelaunaySolverTopologyError>(sum)
            })?
            .map(|sum| sum / neighbor_ids.len() as f64);
        let initial = topology.nodes[index].clone();
        let initial_profile = jacobian::optimization_margin_profile_elements(
            topology,
            incident_elements.iter().copied(),
            cancellation,
        )?;
        let mut best = None;
        for fraction in RELOCATION_FRACTIONS {
            consume_candidate_work(candidate_work, options, cancellation)?;
            topology.nodes[index].coordinates_m = std::array::from_fn(|axis| {
                initial.coordinates_m[axis] * (1.0 - fraction) + target[axis] * fraction
            });
            if corner_signs(topology, &node_index, &incident_elements)? != original_signs {
                topology.nodes[index] = initial.clone();
                continue;
            }
            match jacobian::validate_elements(
                topology,
                incident_elements.iter().copied(),
                maximum_search_work,
                maximum_recursion_depth,
                options.cancellation_check_interval,
                cancellation,
                jacobian_work,
            ) {
                Ok(()) => return Ok(1),
                Err(failure) if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh => {
                    let profile = jacobian::optimization_margin_profile_elements(
                        topology,
                        incident_elements.iter().copied(),
                        cancellation,
                    )?;
                    if profile > initial_profile
                        && best
                            .as_ref()
                            .is_none_or(|(best_profile, _)| profile > *best_profile)
                    {
                        best = Some((profile, topology.nodes[index].coordinates_m));
                    }
                    topology.nodes[index] = initial.clone();
                }
                Err(failure) => {
                    topology.nodes[index] = initial;
                    return Err(failure);
                }
            }
        }
        if let Some((_, coordinates_m)) = best {
            topology.nodes[index].coordinates_m = coordinates_m;
            return Ok(1);
        }
        topology.nodes[index] = initial;
    }
    Ok(0)
}

fn corner_signs(
    topology: &SolverMeshTopology,
    node_index: &BTreeMap<u64, usize>,
    element_indices: &BTreeSet<usize>,
) -> Result<Vec<PredicateSign>, DelaunaySolverTopologyError> {
    element_indices
        .iter()
        .map(|element_index| {
            let element = topology
                .volume_elements
                .get(*element_index)
                .ok_or_else(|| invalid("interior-corner incident element is absent"))?;
            let points = element.node_ids[..4]
                .iter()
                .map(|node_id| {
                    node_index
                        .get(node_id)
                        .map(|index| topology.nodes[*index].coordinates_m)
                        .ok_or_else(|| invalid("interior-corner incident node is absent"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            orient3d(points.try_into().map_err(|_| {
                invalid("interior-corner incident element does not have four corners")
            })?)
            .map_err(|failure| invalid(format!("interior-corner orientation failed: {failure:?}")))
        })
        .collect()
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
