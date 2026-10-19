//! The unbounded conflict is a connected visible hull cap; finite conflict
//! components meet that cap. Search adjacency rather than every tetrahedron.
use super::*;
use std::collections::VecDeque;

pub(super) fn visible_faces(
    topology: &TopologyMutation,
    node: DelaunayVolumeNode,
    seed: Option<(u32, usize)>,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<(u32, usize)>, DelaunayInsertionError> {
    let mut first = seed;
    if first.is_none() {
        for &(index, opposite) in topology.boundary_faces.values() {
            let tet = topology
                .tetrahedron(index)
                .expect("boundary index references active tetrahedron");
            if face_side(topology, tet.vertex_indices, opposite, node, work)? == FaceSide::Exterior
            {
                first = Some((index, opposite));
                break;
            }
        }
    }
    let Some((index, opposite)) = first else {
        return Ok(BTreeSet::new());
    };
    let initial = face(
        topology
            .tetrahedron(index)
            .expect("hull seed is active")
            .vertex_indices,
        opposite,
    );
    let mut pending = VecDeque::from([initial]);
    let mut visited = BTreeSet::new();
    let mut visible = BTreeSet::new();
    while let Some(key) = pending.pop_front() {
        if !visited.insert(key) {
            continue;
        }
        let &(index, opposite) = topology.boundary_faces.get(&key).ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "hull adjacency reached a non-boundary face",
            )
        })?;
        let tet = topology
            .tetrahedron(index)
            .expect("boundary index references active tetrahedron");
        if face_side(topology, tet.vertex_indices, opposite, node, work)? != FaceSide::Exterior {
            continue;
        }
        visible.insert((index, opposite));
        if visible.len() as u64 > work.options.maximum_cavity_boundary_faces {
            return Err(resource(
                "hull insertion visible-face inventory limit exceeded",
            ));
        }
        pending.extend(topology.boundary_neighbors(key));
    }
    Ok(visible)
}

pub(super) fn finite_conflict(
    topology: &TopologyMutation,
    node: DelaunayVolumeNode,
    visible: &BTreeSet<(u32, usize)>,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayInsertionError> {
    let mut pending = visible
        .iter()
        .map(|(index, _)| *index)
        .collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    let mut removed = BTreeSet::new();
    while !pending.is_empty() {
        // Read-only waves on the same topology version. Commit results in the
        // original breadth-first order, preserving the connected conflict policy.
        let mut wave = Vec::with_capacity(256);
        for _ in 0..pending.len().min(256) {
            let index = pending.pop_front().expect("nonempty conflict queue");
            if !visited.insert(index) {
                continue;
            }
            work.predicate()?;
            let cell = topology
                .tetrahedron(index)
                .expect("live conflict adjacency");
            wave.push((index, cell.vertex_indices, cell.neighbors));
        }
        let nodes = topology.nodes();
        let cancellation = work.cancellation();
        let evaluate = |&(index, corners, neighbors): &(u32, [u32; 4], [Option<u32>; 4])| {
            if cancellation.is_cancelled() {
                return Err(error(DelaunayInsertionErrorKind::Cancelled, "cancelled"));
            }
            let relation = super::super::cavity::materialized_relation(
                corners.map(|vertex| nodes[vertex as usize]),
                node,
            )?;
            Ok((index, neighbors, relation))
        };
        let outcomes = if wave.len() >= 32 {
            runmat_cpu_execution::ordered_map(&wave, evaluate)
        } else {
            wave.iter().map(evaluate).collect()
        };
        for outcome in outcomes {
            let (index, neighbors, relation) = outcome?;
            if relation != PredicateSign::Positive {
                continue;
            }
            removed.insert(index);
            if removed.len() as u64 > work.options.maximum_cavity_tetrahedra {
                return Err(resource(
                    "hull insertion conflict tetrahedron limit exceeded",
                ));
            }
            pending.extend(neighbors.into_iter().flatten());
        }
    }
    Ok(removed)
}
