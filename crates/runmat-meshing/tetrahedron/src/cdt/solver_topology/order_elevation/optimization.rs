use super::PreparedEvaluation;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{
    MeshingCancellationSignal, SolverMeshTopology, SolverNodeExactParameter, StableDigest,
};

use super::{geometry, jacobian, MidpointMap};
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind,
    DelaunaySolverTopologyOptions, ProjectionInput,
};

mod patch;
pub(super) use patch::repair_corner_star;
mod relocation;
pub(super) use relocation::relocate_interior_corner;
mod candidates;
mod incidence;
mod indexes;
mod node;
mod repair_state;
mod transaction;
pub(super) use indexes::OptimizationTopology;
use node::NodeOptimizer;
pub(super) use repair_state::RepairState;

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

#[derive(Clone, Copy, Debug)]
struct MovableNode<'a> {
    edge: [u64; 2],
    node_id: u64,
    stable_identity: StableDigest,
    incident_elements: &'a [usize],
    surface_uses: &'a [geometry::SurfaceUse<'a>],
    exact_edge_id: Option<&'a PersistentEntityId>,
}

#[derive(Clone)]
struct Candidate {
    fraction: f64,
    objective: f64,
    coordinates_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
    exact_parameters: Vec<SolverNodeExactParameter>,
}

struct RepairRequest<'a> {
    movable: &'a MovableNode<'a>,
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
    pub counters: &'a super::counters::ElevationCounters,
}

pub(super) struct OptimizationTargets<'a, 'state> {
    pub improve_valid_nodes: bool,
    pub invalid_elements: &'a [usize],
    pub indexes: &'a OptimizationTopology<'a>,
    pub repair: &'a mut RepairState<'state>,
}

pub(super) fn optimize<'source>(
    input: &ProjectionInput<'source>,
    topology: &mut SolverMeshTopology,
    midpoint_by_edge: &MidpointMap<'_>,
    evaluation: PreparedEvaluation<'source, '_>,
    work: OptimizationWork<'_>,
    targets: OptimizationTargets<'_, '_>,
) -> Result<u64, DelaunaySolverTopologyError> {
    let OptimizationTargets {
        improve_valid_nodes,
        invalid_elements,
        indexes,
        repair,
    } = targets;
    let OptimizationWork {
        options,
        cancellation,
        jacobian_work,
        candidate_work,
        counters,
    } = work;
    let scratch_scope = runmat_geometry_core::GeometryAllocationScope::new(evaluation.control);
    let scratch = crate::cdt::solver_topology::storage::Storage(Some(&scratch_scope));
    let movable = movable_nodes(
        topology,
        &indexes.incidence,
        midpoint_by_edge,
        &indexes.surfaces,
        &indexes.edges,
        scratch,
        cancellation,
    )?;
    let rounds = nonoverlapping_rounds(
        movable,
        options.maximum_curved_optimization_rounds,
        topology.volume_elements.len(),
        scratch,
        cancellation,
    )?;
    let mut optimizer = NodeOptimizer {
        input,
        evaluation,
        options,
        cancellation,
        jacobian_work,
        candidate_work,
        improve_valid_nodes,
        invalid_elements,
        counters,
        repair,
    };

    let mut moved = 0_u64;
    for round in rounds {
        for movable in round {
            moved += u64::from(optimizer.optimize(topology, &movable)?);
        }
    }
    Ok(moved)
}

fn movable_nodes<'a>(
    topology: &SolverMeshTopology,
    incidence: &'a incidence::Incidence<'_>,
    midpoint_by_edge: &MidpointMap<'_>,
    surface_uses: &'a geometry::BoundarySurfaceUses<'a>,
    exact_edges: &geometry::BoundaryExactEdges<'a>,
    storage: crate::cdt::solver_topology::storage::Storage<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MovableNode<'a>>, DelaunaySolverTopologyError> {
    let count = midpoint_by_edge.into_iter().count();
    let mut result = storage.vector(count)?;
    for (edge, node_id) in midpoint_by_edge {
        candidate_checkpoint(cancellation)?;
        let incident_elements = incidence.elements(*node_id)?;
        if incident_elements.is_empty() {
            return Err(invalid("curved midside node has no incident Tet10 element"));
        }
        let slot = geometry::node_slot(&topology.nodes, *node_id)?;
        result.push(MovableNode {
            edge: *edge,
            node_id: *node_id,
            stable_identity: topology.nodes[slot].stable_identity,
            incident_elements,
            surface_uses: surface_uses.get(edge).unwrap_or_default(),
            exact_edge_id: exact_edges.get(edge),
        });
    }
    let mut scratch = storage.collect(result.iter().copied())?;
    runmat_cpu_execution::sort_copy_with_checkpoint(
        &mut result,
        &mut scratch,
        |a: &MovableNode<'_>, b: &MovableNode<'_>| a.stable_identity.cmp(&b.stable_identity),
        || candidate_checkpoint(cancellation),
    )?;
    storage.release_backing(scratch)?;
    Ok(result)
}

fn nonoverlapping_rounds<'a>(
    movable: Vec<MovableNode<'a>>,
    maximum_rounds: u32,
    element_count: usize,
    storage: crate::cdt::solver_topology::storage::Storage<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<Vec<MovableNode<'a>>>, DelaunaySolverTopologyError> {
    let words = element_count.div_ceil(64);
    let mut rounds: Vec<(Vec<u64>, Vec<MovableNode<'a>>)> =
        storage.vector(movable.len().min(maximum_rounds as usize))?;
    for node in movable {
        candidate_checkpoint(cancellation)?;
        if let Some((occupied, nodes)) = rounds.iter_mut().find(|(occupied, _)| {
            node.incident_elements
                .iter()
                .all(|element| occupied[*element / 64] & (1u64 << (*element % 64)) == 0)
        }) {
            for &element in node.incident_elements {
                occupied[element / 64] |= 1u64 << (element % 64);
            }
            storage.push(nodes, node)?;
            continue;
        }
        if rounds.len() as u32 >= maximum_rounds {
            return Err(resource(format!(
                "curved-node conflict coloring exceeds its hard round limit of {maximum_rounds}"
            )));
        }
        let mut occupied = storage.vector(words)?;
        occupied.resize(words, 0u64);
        for &element in node.incident_elements {
            occupied[element / 64] |= 1u64 << (element % 64);
        }
        rounds.push((occupied, storage.collect(std::iter::once(node))?));
    }
    let mut result = storage.vector(rounds.len())?;
    for (occupied, nodes) in rounds {
        storage.release_backing(occupied)?;
        result.push(nodes);
    }
    Ok(result)
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

    fn movable(node_id: u64, incident_elements: &[usize]) -> MovableNode<'_> {
        MovableNode {
            edge: [node_id, node_id + 1],
            node_id,
            stable_identity: StableDigest::from_bytes([node_id as u8; 32]),
            incident_elements,
            surface_uses: &[],
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
        let rounds = nonoverlapping_rounds(
            input,
            2,
            3,
            crate::cdt::solver_topology::storage::Storage(None),
            &NeverCancelled,
        )
        .unwrap();
        assert_eq!(
            rounds
                .iter()
                .map(|round| round.iter().map(|node| node.node_id).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 4], vec![3]]
        );
        assert_eq!(
            nonoverlapping_rounds(
                vec![movable(1, &[0]), movable(2, &[0])],
                1,
                1,
                crate::cdt::solver_topology::storage::Storage(None),
                &NeverCancelled,
            )
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
