//! The unbounded conflict is a connected hull cap: the faces the point sees, plus the faces
//! coplanar with it whose circumcircle holds it. Finite conflict components meet that cap.
//! Search adjacency rather than every tetrahedron.
use super::*;
use std::collections::VecDeque;

/// Hull faces in conflict with an exterior point. `visible` faces are seen strictly from
/// outside and receive a new cell. `coplanar` faces lie in a plane through the point and hold it
/// in their circumcircle: the (unbounded) cell over such a face conflicts with the point exactly
/// when the finite cell under it does, since that cell's circumsphere meets the face plane in the
/// face circumcircle. Those finite cells join the cavity, and the cap is connected only through
/// both kinds of face, so the search must cross them (points on a cylinder cap, a box face).
pub(super) struct HullConflict {
    pub(super) visible: BTreeSet<(u32, usize)>,
    pub(super) coplanar: BTreeSet<u32>,
}

pub(super) fn visible_faces(
    topology: &TopologyMutation,
    node: DelaunayVolumeNode,
    seed: Option<(u32, usize)>,
    work: &mut InsertionWork<'_>,
) -> Result<HullConflict, DelaunayInsertionError> {
    let mut first = seed;
    if first.is_none() {
        for &(index, opposite) in topology.boundary_faces.values() {
            let tet = topology
                .tetrahedron(index)
                .expect("boundary index references active tetrahedron");
            if in_conflict(topology, index, tet.vertex_indices, opposite, node, work)?.is_some() {
                first = Some((index, opposite));
                break;
            }
        }
    }
    let mut conflict = HullConflict {
        visible: BTreeSet::new(),
        coplanar: BTreeSet::new(),
    };
    let Some((index, opposite)) = first else {
        return Ok(conflict);
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
        match in_conflict(topology, index, tet.vertex_indices, opposite, node, work)? {
            None => continue,
            Some(FaceSide::Exterior) => {
                conflict.visible.insert((index, opposite));
            }
            Some(_) => {
                conflict.coplanar.insert(index);
            }
        }
        if (conflict.visible.len() + conflict.coplanar.len()) as u64
            > work.options.maximum_cavity_boundary_faces
        {
            return Err(resource(
                "hull insertion visible-face inventory limit exceeded",
            ));
        }
        pending.extend(topology.boundary_neighbors(key));
    }
    Ok(conflict)
}

/// The side of a hull face that is in conflict with `node`, or `None`.
fn in_conflict(
    topology: &TopologyMutation,
    index: u32,
    vertices: [u32; 4],
    opposite: usize,
    node: DelaunayVolumeNode,
    work: &mut InsertionWork<'_>,
) -> Result<Option<FaceSide>, DelaunayInsertionError> {
    Ok(match face_side(topology, vertices, opposite, node, work)? {
        FaceSide::Exterior => Some(FaceSide::Exterior),
        FaceSide::Interior => None,
        FaceSide::Coplanar => {
            work.predicate()?;
            let corners = topology
                .tetrahedron(index)
                .expect("boundary index references active tetrahedron")
                .vertex_indices
                .map(|vertex| topology.nodes()[vertex as usize]);
            (super::super::cavity::materialized_relation(corners, node)? == PredicateSign::Positive)
                .then_some(FaceSide::Coplanar)
        }
    })
}

pub(super) fn finite_conflict(
    topology: &TopologyMutation,
    node: DelaunayVolumeNode,
    conflict: &HullConflict,
    work: &mut InsertionWork<'_>,
) -> Result<BTreeSet<u32>, DelaunayInsertionError> {
    let mut pending = conflict
        .visible
        .iter()
        .map(|(index, _)| *index)
        .chain(conflict.coplanar.iter().copied())
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
