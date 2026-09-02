use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{SolverMeshNode, SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

use super::{consume_candidate_work, geometry, jacobian, OptimizationWork};
use crate::cdt::solver_topology::{
    DelaunayExactEvaluation, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput,
};

const CORNER_STAR_DELTAS: [f64; 9] = [
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
    0.00048828125,
];

pub(in crate::cdt::solver_topology::order_elevation) fn repair_corner_star(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    element_index: usize,
    evaluation: DelaunayExactEvaluation<'_>,
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
        .ok_or_else(|| invalid("corner-star repair element index is out of bounds"))?
        .clone();
    let node_index = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let surface_uses = geometry::boundary_surface_uses(input, topology, cancellation)?;
    let exact_edges = geometry::boundary_exact_edges(topology, cancellation)?;

    for local_corner in 0..4 {
        let local_edges = TETRAHEDRON_MIDSIDE_EDGE_CORNERS
            .iter()
            .enumerate()
            .filter_map(|(local_edge, edge)| edge.contains(&local_corner).then_some(local_edge))
            .collect::<Vec<_>>();
        let midpoint_node_ids = local_edges
            .iter()
            .map(|local_edge| element.node_ids[4 + local_edge])
            .collect::<Vec<_>>();
        let incident_elements = topology
            .volume_elements
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                midpoint_node_ids
                    .iter()
                    .any(|node_id| candidate.node_ids.contains(node_id))
                    .then_some(index)
            })
            .collect::<BTreeSet<_>>();
        let initial_profile = jacobian::optimization_margin_profile_elements(
            topology,
            incident_elements.iter().copied(),
            cancellation,
        )?;
        let baseline = midpoint_node_ids
            .iter()
            .map(|node_id| {
                let index = *node_index
                    .get(node_id)
                    .ok_or_else(|| invalid("corner-star midside node is absent"))?;
                Ok((index, topology.nodes[index].clone()))
            })
            .collect::<Result<Vec<_>, DelaunaySolverTopologyError>>()?;

        for toward_corner in [true, false] {
            for delta in CORNER_STAR_DELTAS {
                restore(topology, &baseline);
                let mut complete = true;
                for &local_edge in &local_edges {
                    let local_endpoints = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[local_edge];
                    let mut endpoint_ids = [
                        element.node_ids[local_endpoints[0]],
                        element.node_ids[local_endpoints[1]],
                    ];
                    endpoint_ids.sort_unstable();
                    let corner_node_id = element.node_ids[local_corner];
                    let corner_is_left = endpoint_ids[0] == corner_node_id;
                    let toward_left = corner_is_left == toward_corner;
                    let fraction = if toward_left {
                        0.5 - delta
                    } else {
                        0.5 + delta
                    };
                    let left = topology.nodes[*node_index
                        .get(&endpoint_ids[0])
                        .ok_or_else(|| invalid("corner-star left endpoint is absent"))?]
                    .clone();
                    let right = topology.nodes[*node_index
                        .get(&endpoint_ids[1])
                        .ok_or_else(|| invalid("corner-star right endpoint is absent"))?]
                    .clone();
                    let edge = super::super::sorted_edge(endpoint_ids);
                    consume_candidate_work(candidate_work, options, cancellation)?;
                    let Some(point) = geometry::edge_geometry(
                        input,
                        evaluation,
                        options,
                        geometry::EdgeGeometryRequest {
                            left: &left,
                            right: &right,
                            exact_edge_id: exact_edges.get(&edge),
                            surface_uses: surface_uses.get(&edge).map_or(&[], Vec::as_slice),
                            fraction,
                        },
                    )?
                    else {
                        complete = false;
                        break;
                    };
                    let midpoint_node_id = element.node_ids[4 + local_edge];
                    let midpoint_index = *node_index
                        .get(&midpoint_node_id)
                        .ok_or_else(|| invalid("corner-star midside node is absent"))?;
                    topology.nodes[midpoint_index].coordinates_m = point.coordinates_m;
                    topology.nodes[midpoint_index].exact_parameters = point.exact_parameters;
                }
                if !complete {
                    continue;
                }
                match jacobian::validate_elements(
                    topology,
                    incident_elements.iter().copied(),
                    input.request.resources.maximum_search_work,
                    input.request.resources.maximum_recursion_depth,
                    options.cancellation_check_interval,
                    cancellation,
                    jacobian_work,
                ) {
                    Ok(()) => return Ok(changed_count(topology, &baseline)),
                    Err(failure)
                        if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh =>
                    {
                        let profile = jacobian::optimization_margin_profile_elements(
                            topology,
                            incident_elements.iter().copied(),
                            cancellation,
                        )?;
                        if profile > initial_profile {
                            return Ok(changed_count(topology, &baseline));
                        }
                    }
                    Err(failure) => {
                        restore(topology, &baseline);
                        return Err(failure);
                    }
                }
            }
        }
        restore(topology, &baseline);
    }
    Ok(0)
}

fn changed_count(topology: &SolverMeshTopology, baseline: &[(usize, SolverMeshNode)]) -> u64 {
    baseline
        .iter()
        .filter(|(index, node)| topology.nodes[*index] != *node)
        .count() as u64
}

fn restore(topology: &mut SolverMeshTopology, baseline: &[(usize, SolverMeshNode)]) {
    for (index, node) in baseline {
        topology.nodes[*index] = node.clone();
    }
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    super::error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
