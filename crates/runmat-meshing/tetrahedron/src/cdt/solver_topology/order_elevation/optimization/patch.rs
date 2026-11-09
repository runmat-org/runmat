use super::super::PreparedEvaluation;
use super::{
    consume_candidate_work, geometry, jacobian, transaction, OptimizationTopology, OptimizationWork,
};
use crate::cdt::solver_topology::{
    storage::Storage, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, ProjectionInput,
};
use runmat_geometry_core::GeometryAllocationScope;
use runmat_meshing_core::{SolverMeshTopology, TETRAHEDRON_MIDSIDE_EDGE_CORNERS};

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
    input: &ProjectionInput<'_>,
    topology: &mut SolverMeshTopology,
    element_index: usize,
    evaluation: PreparedEvaluation<'_, '_>,
    indexes: &OptimizationTopology<'_>,
    work: OptimizationWork<'_>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let OptimizationWork {
        options,
        cancellation,
        jacobian_work,
        candidate_work,
        counters: _,
    } = work;
    let element = topology
        .volume_elements
        .get(element_index)
        .ok_or_else(|| invalid("corner-star repair element index is out of bounds"))?;
    let nodes: [u64; 10] = element
        .node_ids
        .as_slice()
        .try_into()
        .map_err(|_| invalid("corner-star repair requires ten nodes"))?;
    for local_corner in 0..4 {
        let scope = GeometryAllocationScope::new(evaluation.control);
        let storage = Storage(Some(&scope));
        let mut local_edges = [0usize; 3];
        let mut count = 0;
        for (edge, endpoints) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            if endpoints.contains(&local_corner) {
                local_edges[count] = edge;
                count += 1;
            }
        }
        let midpoint_ids = local_edges.map(|edge| nodes[4 + edge]);
        let mut slots = [0usize; 3];
        for (slot, id) in slots.iter_mut().zip(midpoint_ids) {
            *slot = geometry::node_slot(&topology.nodes, id)?;
        }
        let mut incidence_count = 0usize;
        for id in midpoint_ids {
            incidence_count = incidence_count
                .checked_add(indexes.incidence.elements(id)?.len())
                .ok_or_else(|| super::resource("corner-star incidence storage overflow"))?;
        }
        let mut incident = storage.vector(incidence_count)?;
        for id in midpoint_ids {
            incident.extend_from_slice(indexes.incidence.elements(id)?);
        }
        incident.sort_unstable();
        incident.dedup();
        let initial_profile = jacobian::optimization_margin_profile_elements(
            topology,
            incident.iter().copied(),
            cancellation,
            storage,
        )?;
        for toward_corner in [true, false] {
            for delta in CORNER_STAR_DELTAS {
                let moved = transaction::edit_nodes(
                    topology,
                    &slots,
                    evaluation.control,
                    |topology, control| {
                        let evaluation = PreparedEvaluation {
                            control,
                            ..evaluation
                        };
                        for &local_edge in &local_edges {
                            let endpoints = TETRAHEDRON_MIDSIDE_EDGE_CORNERS[local_edge];
                            let mut endpoint_ids = [nodes[endpoints[0]], nodes[endpoints[1]]];
                            endpoint_ids.sort_unstable();
                            let toward_left =
                                (endpoint_ids[0] == nodes[local_corner]) == toward_corner;
                            let fraction = if toward_left {
                                0.5 - delta
                            } else {
                                0.5 + delta
                            };
                            let left = &topology.nodes
                                [geometry::node_slot(&topology.nodes, endpoint_ids[0])?];
                            let right = &topology.nodes
                                [geometry::node_slot(&topology.nodes, endpoint_ids[1])?];
                            let edge = super::super::sorted_edge(endpoint_ids);
                            consume_candidate_work(candidate_work, options, cancellation)?;
                            let Some(point) = geometry::edge_geometry(
                                input,
                                evaluation,
                                options,
                                geometry::EdgeGeometryRequest {
                                    left,
                                    right,
                                    exact_edge_id: indexes.edges.get(&edge),
                                    surface_uses: indexes.surfaces.get(&edge).unwrap_or_default(),
                                    fraction,
                                },
                            )?
                            else {
                                return Ok(false);
                            };
                            let slot = geometry::node_slot(&topology.nodes, nodes[4 + local_edge])?;
                            topology.nodes[slot].coordinates_m = runmat_types::physical::Quantity::<
                                runmat_types::physical::kind::Position,
                                _,
                            >::new(
                                point.coordinates_m.canonical_value(),
                            );
                            topology.nodes[slot].exact_parameters = point.exact_parameters;
                        }
                        match jacobian::validate_elements(
                            topology,
                            incident.iter().copied(),
                            jacobian::ELEMENT_SEARCH_WORK,
                            input.request.limits.maximum_recursion_depth,
                            options.cancellation_check_interval,
                            cancellation,
                            jacobian::ValidationWork {
                                counter: jacobian_work,
                                storage: Storage(Some(control)),
                            },
                        ) {
                            Ok(()) => Ok(true),
                            Err(failure)
                                if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh =>
                            {
                                let profile = jacobian::optimization_margin_profile_elements(
                                    topology,
                                    incident.iter().copied(),
                                    cancellation,
                                    Storage(Some(control)),
                                )?;
                                Ok(profile > initial_profile)
                            }
                            Err(failure) => Err(failure),
                        }
                    },
                )?;
                if moved > 0 {
                    return Ok(moved);
                }
            }
        }
    }
    Ok(0)
}
fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    super::error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
