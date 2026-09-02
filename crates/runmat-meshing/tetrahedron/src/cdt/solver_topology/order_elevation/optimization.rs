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

mod patch;
pub(super) use patch::repair_corner_star;
mod relocation;
pub(super) use relocation::relocate_interior_corner;
mod candidates;

// These fractions deliberately exclude the initial parameter midpoint. They form a
// stable, bounded search set and preserve the source edge's closed parameter domain.
const SHAPE_CANDIDATE_FRACTIONS: [f64; 6] = [0.125, 0.25, 0.375, 0.625, 0.75, 0.875];
const REPAIR_CANDIDATE_FRACTIONS: [f64; 24] = [
    0.499755859375,
    0.500244140625,
    0.49951171875,
    0.50048828125,
    0.4990234375,
    0.5009765625,
    0.498046875,
    0.501953125,
    0.49609375,
    0.50390625,
    0.4921875,
    0.5078125,
    0.484375,
    0.515625,
    0.46875,
    0.53125,
    0.4375,
    0.5625,
    0.375,
    0.625,
    0.25,
    0.75,
    0.125,
    0.875,
];

#[derive(Clone, Debug)]
struct MovableNode {
    edge: [u64; 2],
    node_id: u64,
    stable_identity: StableDigest,
    incident_elements: Vec<usize>,
    surface_uses: Vec<geometry::SurfaceUse>,
    exact_edge_id: Option<PersistentEntityId>,
}

#[derive(Clone)]
struct Candidate {
    fraction: f64,
    objective: f64,
    coordinates_m: [f64; 3],
    exact_parameters: Vec<SolverNodeExactParameter>,
}

struct RepairRequest<'a> {
    movable: &'a MovableNode,
    node_index: usize,
    endpoints: [runmat_meshing_core::SolverMeshNode; 2],
    target: [f64; 3],
    initial: runmat_meshing_core::SolverMeshNode,
}

pub(in crate::cdt::solver_topology::order_elevation) struct OptimizationWork<'a> {
    pub options: DelaunaySolverTopologyOptions,
    pub cancellation: &'a dyn MeshingCancellationSignal,
    pub jacobian_work: &'a mut u64,
    pub candidate_work: &'a mut u64,
}

pub(super) fn optimize(
    input: &DelaunaySolverTopologyInput<'_>,
    topology: &mut SolverMeshTopology,
    midpoint_by_edge: &MidpointMap,
    evaluation: DelaunayExactEvaluation<'_>,
    work: OptimizationWork<'_>,
    improve_valid_nodes: bool,
    invalid_elements: &BTreeSet<usize>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let OptimizationWork {
        options,
        cancellation,
        jacobian_work,
        candidate_work,
    } = work;
    let surface_uses = geometry::boundary_surface_uses(input, topology, cancellation)?;
    let exact_edges = geometry::boundary_exact_edges(topology, cancellation)?;
    let movable = movable_nodes(
        topology,
        midpoint_by_edge,
        &surface_uses,
        &exact_edges,
        cancellation,
    )?;
    let rounds = nonoverlapping_rounds(
        movable,
        options.maximum_curved_optimization_rounds,
        cancellation,
    )?;
    let mut node_index = BTreeMap::new();
    for (index, node) in topology.nodes.iter().enumerate() {
        candidate_checkpoint(cancellation)?;
        node_index.insert(node.node_id, index);
    }
    let mut optimizer = NodeOptimizer {
        input,
        evaluation,
        options,
        cancellation,
        jacobian_work,
        candidate_work,
        improve_valid_nodes,
        invalid_elements,
        node_index,
    };

    let mut moved = 0_u64;
    for round in rounds {
        for movable in round {
            moved += u64::from(optimizer.optimize(topology, &movable)?);
        }
    }
    Ok(moved)
}

fn movable_nodes(
    topology: &SolverMeshTopology,
    midpoint_by_edge: &MidpointMap,
    surface_uses: &BTreeMap<[u64; 2], Vec<geometry::SurfaceUse>>,
    exact_edges: &BTreeMap<[u64; 2], PersistentEntityId>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MovableNode>, DelaunaySolverTopologyError> {
    let mut incident_by_node = BTreeMap::<u64, Vec<usize>>::new();
    for (element_index, element) in topology.volume_elements.iter().enumerate() {
        candidate_checkpoint(cancellation)?;
        for node_id in element.node_ids.iter().skip(4) {
            incident_by_node
                .entry(*node_id)
                .or_default()
                .push(element_index);
        }
    }
    let mut identity_by_node = BTreeMap::new();
    for node in &topology.nodes {
        candidate_checkpoint(cancellation)?;
        identity_by_node.insert(node.node_id, node.stable_identity);
    }
    let mut result = Vec::new();
    for (edge, node_id) in midpoint_by_edge {
        candidate_checkpoint(cancellation)?;
        let uses = surface_uses.get(edge).cloned().unwrap_or_default();
        let exact_edge_id = exact_edges.get(edge).cloned();
        let incident_elements = incident_by_node.remove(node_id).unwrap_or_default();
        if incident_elements.is_empty() {
            return Err(invalid("curved midside node has no incident Tet10 element"));
        }
        let stable_identity = *identity_by_node
            .get(node_id)
            .ok_or_else(|| invalid("curved midside node is absent"))?;
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
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<Vec<MovableNode>>, DelaunaySolverTopologyError> {
    let mut rounds = Vec::<(BTreeSet<usize>, Vec<MovableNode>)>::new();
    for node in movable {
        candidate_checkpoint(cancellation)?;
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
    improve_valid_nodes: bool,
    invalid_elements: &'b BTreeSet<usize>,
    node_index: BTreeMap<u64, usize>,
}

impl NodeOptimizer<'_, '_> {
    fn optimize(
        &mut self,
        topology: &mut SolverMeshTopology,
        movable: &MovableNode,
    ) -> Result<bool, DelaunaySolverTopologyError> {
        let node_index = *self
            .node_index
            .get(&movable.node_id)
            .ok_or_else(|| invalid("curved midside node is absent"))?;
        let left = topology
            .nodes
            .get(
                *self
                    .node_index
                    .get(&movable.edge[0])
                    .ok_or_else(|| invalid("curved edge endpoint is absent"))?,
            )
            .ok_or_else(|| invalid("curved edge endpoint is absent"))?
            .clone();
        let right = topology
            .nodes
            .get(
                *self
                    .node_index
                    .get(&movable.edge[1])
                    .ok_or_else(|| invalid("curved edge endpoint is absent"))?,
            )
            .ok_or_else(|| invalid("curved edge endpoint is absent"))?
            .clone();
        let target = midpoint(left.coordinates_m, right.coordinates_m);
        let initial = topology.nodes[node_index].clone();
        let initial_objective = squared_distance(initial.coordinates_m, target);
        let initial_is_valid = movable
            .incident_elements
            .iter()
            .all(|element| !self.invalid_elements.contains(element));
        if initial_is_valid && !self.improve_valid_nodes {
            return Ok(false);
        }
        if !initial_is_valid {
            return self.repair_invalid(
                topology,
                RepairRequest {
                    movable,
                    node_index,
                    endpoints: [left, right],
                    target,
                    initial,
                },
            );
        }
        let mut candidates = Vec::new();

        for fraction in SHAPE_CANDIDATE_FRACTIONS {
            if let Some(candidate) = self.candidate(movable, &left, &right, target, fraction)? {
                if candidate.objective < initial_objective {
                    candidates.push(candidate);
                }
            }
        }
        candidates.sort_by(|left, right| {
            left.objective
                .total_cmp(&right.objective)
                .then_with(|| left.fraction.total_cmp(&right.fraction))
        });

        for candidate in candidates {
            topology.nodes[node_index].coordinates_m = candidate.coordinates_m;
            topology.nodes[node_index].exact_parameters = candidate.exact_parameters.clone();
            match jacobian::validate_elements(
                topology,
                movable.incident_elements.iter().copied(),
                self.input.request.resources.maximum_search_work,
                self.input.request.resources.maximum_recursion_depth,
                self.options.cancellation_check_interval,
                self.cancellation,
                self.jacobian_work,
            ) {
                Ok(()) => return Ok(topology.nodes[node_index] != initial),
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
        Ok(false)
    }

    fn repair_invalid(
        &mut self,
        topology: &mut SolverMeshTopology,
        request: RepairRequest<'_>,
    ) -> Result<bool, DelaunaySolverTopologyError> {
        let initial_profile = jacobian::optimization_margin_profile_elements(
            topology,
            request.movable.incident_elements.iter().copied(),
            self.cancellation,
        )?;
        let mut best_candidate = None::<(Vec<u64>, Candidate)>;
        let mut candidates = Vec::new();
        for fraction in REPAIR_CANDIDATE_FRACTIONS {
            let Some(candidate) = self.candidate(
                request.movable,
                &request.endpoints[0],
                &request.endpoints[1],
                request.target,
                fraction,
            )?
            else {
                continue;
            };
            candidates.push(candidate);
        }
        if request.movable.exact_edge_id.is_none() && request.movable.surface_uses.is_empty() {
            for (index, coordinates_m) in candidates::interior_coordinates(
                request.initial.coordinates_m,
                request.endpoints.clone().map(|node| node.coordinates_m),
            )
            .into_iter()
            .enumerate()
            {
                consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
                candidates.push(Candidate {
                    fraction: 2.0 + index as f64,
                    objective: squared_distance(coordinates_m, request.target),
                    coordinates_m,
                    exact_parameters: Vec::new(),
                });
            }
        } else if request.movable.exact_edge_id.is_none() && request.movable.surface_uses.len() == 1
        {
            let surface_use = &request.movable.surface_uses[0];
            for (index, evaluator_uv) in candidates::surface_parameters(
                &request.initial,
                [&request.endpoints[0], &request.endpoints[1]],
                surface_use,
            )
            .into_iter()
            .enumerate()
            {
                consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
                let Some(point) = geometry::surface_geometry(
                    self.input,
                    self.evaluation,
                    self.options,
                    surface_use,
                    evaluator_uv,
                )?
                else {
                    continue;
                };
                candidates.push(Candidate {
                    fraction: 2.0 + index as f64,
                    objective: squared_distance(point.coordinates_m, request.target),
                    coordinates_m: point.coordinates_m,
                    exact_parameters: point.exact_parameters,
                });
            }
        }
        for candidate in candidates {
            topology.nodes[request.node_index].coordinates_m = candidate.coordinates_m;
            topology.nodes[request.node_index].exact_parameters =
                candidate.exact_parameters.clone();
            let profile = jacobian::optimization_margin_profile_elements(
                topology,
                request.movable.incident_elements.iter().copied(),
                self.cancellation,
            )?;
            if profile_improves(
                &initial_profile,
                best_candidate
                    .as_ref()
                    .map(|(profile, _)| profile.as_slice()),
                &profile,
            ) {
                best_candidate = Some((profile, candidate));
            }
            topology.nodes[request.node_index] = request.initial.clone();
        }
        if let Some((_, candidate)) = best_candidate {
            topology.nodes[request.node_index].coordinates_m = candidate.coordinates_m;
            topology.nodes[request.node_index].exact_parameters = candidate.exact_parameters;
            return Ok(topology.nodes[request.node_index] != request.initial);
        }
        topology.nodes[request.node_index] = request.initial;
        Ok(false)
    }

    fn candidate(
        &mut self,
        movable: &MovableNode,
        left: &runmat_meshing_core::SolverMeshNode,
        right: &runmat_meshing_core::SolverMeshNode,
        target: [f64; 3],
        fraction: f64,
    ) -> Result<Option<Candidate>, DelaunaySolverTopologyError> {
        consume_candidate_work(self.candidate_work, self.options, self.cancellation)?;
        Ok(geometry::edge_geometry(
            self.input,
            self.evaluation,
            self.options,
            geometry::EdgeGeometryRequest {
                left,
                right,
                exact_edge_id: movable.exact_edge_id.as_ref(),
                surface_uses: &movable.surface_uses,
                fraction,
            },
        )?
        .and_then(|point| {
            let objective = squared_distance(point.coordinates_m, target);
            objective.is_finite().then_some(Candidate {
                fraction,
                objective,
                coordinates_m: point.coordinates_m,
                exact_parameters: point.exact_parameters,
            })
        }))
    }
}

fn profile_improves(initial: &[u64], best: Option<&[u64]>, candidate: &[u64]) -> bool {
    candidate > initial && best.is_none_or(|best| candidate > best)
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
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    if cancellation.is_cancelled() {
        return Err(cancelled());
    }
    Ok(())
}

fn consume_candidate_work(
    work: &mut u64,
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunaySolverTopologyError> {
    *work = work
        .checked_add(1)
        .ok_or_else(|| resource("curved-node candidate counter overflowed"))?;
    if *work > options.maximum_curved_optimization_candidates {
        return Err(resource(format!(
            "curved-node optimization exceeds its hard candidate limit of {}",
            options.maximum_curved_optimization_candidates
        )));
    }
    candidate_checkpoint(cancellation)
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
        let rounds = nonoverlapping_rounds(input, 2, &NeverCancelled).unwrap();
        assert_eq!(
            rounds
                .iter()
                .map(|round| round.iter().map(|node| node.node_id).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 4], vec![3]]
        );
        assert_eq!(
            nonoverlapping_rounds(vec![movable(1, &[0]), movable(2, &[0])], 1, &NeverCancelled,)
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

        assert_eq!(
            candidate_checkpoint(&Cancelled).unwrap_err().kind,
            DelaunaySolverTopologyErrorKind::Cancelled
        );
        candidate_checkpoint(&NeverCancelled).unwrap();
    }

    #[test]
    fn repair_profile_requires_a_strict_lexicographic_improvement() {
        let initial = [10, 20, 30];
        assert!(profile_improves(&initial, None, &[10, 21, 0]));
        assert!(!profile_improves(&initial, None, &[10, 20, 30]));
        assert!(!profile_improves(&initial, None, &[9, u64::MAX, u64::MAX]));
        assert!(profile_improves(&initial, Some(&[10, 21, 0]), &[10, 22, 0]));
        assert!(!profile_improves(
            &initial,
            Some(&[10, 22, 0]),
            &[10, 21, u64::MAX]
        ));
    }
}
