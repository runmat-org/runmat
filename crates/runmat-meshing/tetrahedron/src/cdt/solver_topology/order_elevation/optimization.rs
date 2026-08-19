use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    MeshingCancellationSignal, SolverMeshTopology, SolverNodeExactParameter, StableDigest,
};

use super::{geometry, jacobian, MidpointMap};
use crate::cdt::solver_topology::{
    error, DelaunayExactEvaluation, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyInput, DelaunaySolverTopologyOptions,
};

// These fractions deliberately exclude the initial parameter midpoint. They form a
// stable, bounded search set and preserve the source edge's closed parameter domain.
const CANDIDATE_FRACTIONS: [f64; 6] = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875];

#[derive(Clone, Debug)]
struct MovableNode {
    edge: [u64; 2],
    node_id: u64,
    stable_identity: StableDigest,
    incident_elements: Vec<usize>,
    surface_uses: Vec<geometry::SurfaceUse>,
    exact_edge_id: Option<PersistentEntityId>,
}

struct Candidate {
    fraction: f64,
    objective: f64,
    coordinates_m: [f64; 3],
    exact_parameters: Vec<SolverNodeExactParameter>,
}

pub(super) fn optimize(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    midpoint_by_edge: &MidpointMap,
    evaluation: DelaunayExactEvaluation<'_>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
    jacobian_work: &mut u64,
) -> Result<(), DelaunaySolverTopologyError> {
    let surface_uses = geometry::boundary_surface_uses(input, topology)?;
    let exact_edges = geometry::boundary_exact_edges(topology)?;
    let movable = movable_nodes(topology, midpoint_by_edge, &surface_uses, &exact_edges)?;
    let rounds = nonoverlapping_rounds(movable, options.maximum_curved_optimization_rounds)?;
    let mut candidate_work = 0_u64;
    let mut optimizer = NodeOptimizer {
        input,
        evaluation,
        options,
        cancellation,
        jacobian_work,
        candidate_work: &mut candidate_work,
    };

    for round in rounds {
        for movable in round {
            optimizer.optimize(topology, &movable)?;
        }
    }
    Ok(())
}

fn movable_nodes(
    topology: &SolverMeshTopology,
    midpoint_by_edge: &MidpointMap,
    surface_uses: &BTreeMap<[u64; 2], Vec<geometry::SurfaceUse>>,
    exact_edges: &BTreeMap<[u64; 2], PersistentEntityId>,
) -> Result<Vec<MovableNode>, DelaunaySolverTopologyError> {
    let mut result = Vec::new();
    for (edge, node_id) in midpoint_by_edge {
        let uses = surface_uses.get(edge).cloned().unwrap_or_default();
        let exact_edge_id = exact_edges.get(edge).cloned();
        if uses.is_empty() && exact_edge_id.is_none() {
            continue;
        }
        let incident_elements = topology
            .volume_elements
            .iter()
            .enumerate()
            .filter_map(|(index, element)| element.node_ids.contains(node_id).then_some(index))
            .collect::<Vec<_>>();
        if incident_elements.is_empty() {
            return Err(invalid("curved midside node has no incident Tet10 element"));
        }
        let stable_identity = topology
            .nodes
            .iter()
            .find(|node| node.node_id == *node_id)
            .ok_or_else(|| invalid("curved midside node is absent"))?
            .stable_identity;
        result.push(MovableNode {
            edge: *edge,
            node_id: *node_id,
            stable_identity,
            incident_elements,
            surface_uses: uses,
            exact_edge_id,
        });
    }
    result.sort_by_key(|node| node.stable_identity);
    Ok(result)
}

fn nonoverlapping_rounds(
    movable: Vec<MovableNode>,
    maximum_rounds: u32,
) -> Result<Vec<Vec<MovableNode>>, DelaunaySolverTopologyError> {
    let mut rounds = Vec::<(BTreeSet<usize>, Vec<MovableNode>)>::new();
    for node in movable {
        if let Some((occupied, nodes)) = rounds.iter_mut().find(|(occupied, _)| {
            node.incident_elements
                .iter()
                .all(|element| !occupied.contains(element))
        }) {
            occupied.extend(node.incident_elements.iter().copied());
            nodes.push(node);
            continue;
        }
        if rounds.len() as u32 >= maximum_rounds {
            return Err(resource(format!(
                "curved-node conflict coloring exceeds its hard round limit of {maximum_rounds}"
            )));
        }
        rounds.push((node.incident_elements.iter().copied().collect(), vec![node]));
    }
    Ok(rounds.into_iter().map(|(_, nodes)| nodes).collect())
}

struct NodeOptimizer<'a, 'b> {
    input: &'a DelaunaySolverTopologyInput<'a>,
    evaluation: DelaunayExactEvaluation<'a>,
    options: DelaunaySolverTopologyOptions,
    cancellation: &'b dyn MeshingCancellationSignal,
    jacobian_work: &'b mut u64,
    candidate_work: &'b mut u64,
}

impl NodeOptimizer<'_, '_> {
    fn optimize(
        &mut self,
        topology: &mut SolverMeshTopology,
        movable: &MovableNode,
    ) -> Result<(), DelaunaySolverTopologyError> {
        let node_index = topology
            .nodes
            .iter()
            .position(|node| node.node_id == movable.node_id)
            .ok_or_else(|| invalid("curved midside node is absent"))?;
        let left = topology
            .nodes
            .iter()
            .find(|node| node.node_id == movable.edge[0])
            .ok_or_else(|| invalid("curved edge endpoint is absent"))?
            .clone();
        let right = topology
            .nodes
            .iter()
            .find(|node| node.node_id == movable.edge[1])
            .ok_or_else(|| invalid("curved edge endpoint is absent"))?
            .clone();
        let target = midpoint(left.coordinates_m, right.coordinates_m);
        let initial = topology.nodes[node_index].clone();
        let initial_objective = squared_distance(initial.coordinates_m, target);
        let mut candidates = Vec::new();

        for fraction in CANDIDATE_FRACTIONS {
            *self.candidate_work = self
                .candidate_work
                .checked_add(1)
                .ok_or_else(|| resource("curved-node candidate counter overflowed"))?;
            if *self.candidate_work > self.options.maximum_curved_optimization_candidates {
                return Err(resource(format!(
                    "curved-node optimization exceeds its hard candidate limit of {}",
                    self.options.maximum_curved_optimization_candidates
                )));
            }
            candidate_checkpoint(
                *self.candidate_work,
                self.options.cancellation_check_interval,
                self.cancellation,
            )?;
            let Some(point) = geometry::edge_geometry(
                self.input,
                self.evaluation,
                self.options,
                geometry::EdgeGeometryRequest {
                    left: &left,
                    right: &right,
                    exact_edge_id: movable.exact_edge_id.as_ref(),
                    surface_uses: &movable.surface_uses,
                    fraction,
                },
            )?
            else {
                continue;
            };
            let objective = squared_distance(point.coordinates_m, target);
            if objective.is_finite() && objective < initial_objective {
                candidates.push(Candidate {
                    fraction,
                    objective,
                    coordinates_m: point.coordinates_m,
                    exact_parameters: point.exact_parameters,
                });
            }
        }
        candidates.sort_by(|left, right| {
            left.objective
                .total_cmp(&right.objective)
                .then_with(|| left.fraction.total_cmp(&right.fraction))
        });

        for candidate in candidates {
            topology.nodes[node_index].coordinates_m = candidate.coordinates_m;
            topology.nodes[node_index].exact_parameters = candidate.exact_parameters;
            match jacobian::validate_elements(
                topology,
                movable.incident_elements.iter().copied(),
                self.input.request.resources.maximum_search_work,
                self.input.request.resources.maximum_recursion_depth,
                self.options.cancellation_check_interval,
                self.cancellation,
                self.jacobian_work,
            ) {
                Ok(()) => return Ok(()),
                Err(failure) if failure.kind == DelaunaySolverTopologyErrorKind::InvalidMesh => {
                    topology.nodes[node_index] = initial.clone();
                }
                Err(failure) => {
                    topology.nodes[node_index] = initial;
                    return Err(failure);
                }
            }
        }
        topology.nodes[node_index] = initial;
        Ok(())
    }
}

fn midpoint(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] * 0.5 + right[axis] * 0.5)
}

fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum()
}

fn invalid(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}

fn resource(reason: impl Into<String>) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::ResourceLimit, reason)
}

fn cancelled() -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::Cancelled, "cancelled")
}

fn candidate_checkpoint(
    work: u64,
    interval: u64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    if work.is_multiple_of(interval) && cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_meshing_core::NeverCancelled;

    fn movable(node_id: u64, incident_elements: &[usize]) -> MovableNode {
        MovableNode {
            edge: [node_id, node_id + 1],
            node_id,
            stable_identity: StableDigest::from_bytes([node_id as u8; 32]),
            incident_elements: incident_elements.to_vec(),
            surface_uses: Vec::new(),
            exact_edge_id: None,
        }
    }

    #[test]
    fn coloring_is_stable_and_never_overlaps_incident_elements() {
        let input = vec![
            movable(1, &[0]),
            movable(2, &[1]),
            movable(3, &[0, 1]),
            movable(4, &[2]),
        ];
        let rounds = nonoverlapping_rounds(input, 2).unwrap();
        assert_eq!(
            rounds
                .iter()
                .map(|round| round.iter().map(|node| node.node_id).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 4], vec![3]]
        );
        assert_eq!(
            nonoverlapping_rounds(vec![movable(1, &[0]), movable(2, &[0])], 1)
                .unwrap_err()
                .kind,
            DelaunaySolverTopologyErrorKind::ResourceLimit
        );
    }

    #[test]
    fn candidate_checkpoint_has_bounded_cancellation_latency() {
        struct Cancelled;
        impl MeshingCancellationSignal for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }

        candidate_checkpoint(1, 2, &Cancelled).unwrap();
        assert_eq!(
            candidate_checkpoint(2, 2, &Cancelled).unwrap_err().kind,
            DelaunaySolverTopologyErrorKind::Cancelled
        );
        candidate_checkpoint(2, 2, &NeverCancelled).unwrap();
    }
}
