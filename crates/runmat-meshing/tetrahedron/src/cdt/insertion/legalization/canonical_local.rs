//! Canonical two-to-three legalization over one maintained owner. Queries retain
//! the legacy cell-key/opposite order and symbolic sphere policy. A two-to-three
//! flip preserves old edges and adds its apex edge; only changed adjacencies can
//! make a previously clean face geometrically illegal. Connectivity is rechecked
//! on every popped query. Independent whole-topology admission remains at freeze.
use super::*;
use crate::cdt::{insertion::topology_access::TetrahedralTopology, mutation::TopologyMutation};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct FaceQuery {
    cell_key: [u32; 4],
    opposite: usize,
    cell: u32,
}
fn key(mut vertices: [u32; 4]) -> [u32; 4] {
    vertices.sort_unstable();
    vertices
}
fn enqueue(
    topology: &TopologyMutation,
    cell: u32,
    focus: Option<u32>,
    pending: &mut BTreeSet<FaceQuery>,
) {
    let Some(tetrahedron) = topology.tetrahedron(cell) else {
        return;
    };
    if focus.is_some_and(|focus| !tetrahedron.vertex_indices.contains(&focus)) {
        return;
    }
    for opposite in 0..4 {
        if tetrahedron.neighbors[opposite].is_some()
            && focus.is_none_or(|focus| tetrahedron.vertex_indices[opposite] == focus)
        {
            pending.insert(FaceQuery {
                cell_key: key(tetrahedron.vertex_indices),
                opposite,
                cell,
            });
        }
    }
}

pub(super) fn restore(
    topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus_identity: Option<StableDigest>,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut topology = TopologyMutation::from_canonical(topology);
    let focus = focus_identity
        .map(|identity| {
            topology.node_index(identity).ok_or_else(|| {
                error(
                    DelaunayInsertionErrorKind::InvalidTopology,
                    "local legalization focus node is absent",
                )
            })
        })
        .transpose()?;
    let mut pending = BTreeSet::new();
    for cell in 0..topology.tetrahedron_slot_count() {
        work.checkpoint()?;
        enqueue(&topology, cell as u32, focus, &mut pending);
    }
    while let Some(query) = pending.pop_first() {
        let Some((right_index, face)) = candidate(
            &topology,
            query,
            protected_faces,
            protect_region_boundaries,
            focus,
            work,
        )?
        else {
            continue;
        };
        let left = topology.tetrahedron(query.cell).expect("live face owner");
        let right = topology
            .tetrahedron(right_index)
            .expect("live face neighbor");
        if left.region_id != right.region_id {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "Delaunay legalization cannot cross a region boundary",
            ));
        }
        let region = left.region_id.clone();
        let left_apex = left.vertex_indices[query.opposite];
        let right_apex = right
            .vertex_indices
            .into_iter()
            .find(|vertex| !face.contains(vertex))
            .expect("distinct face apex");
        let created = [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]].map(|edge| {
            (
                key([left_apex, right_apex, edge[0], edge[1]]),
                region.clone(),
            )
        });
        let count = topology
            .active_tetrahedron_count()
            .checked_add(1)
            .ok_or_else(|| super::super::resource("legalization cell count overflow"))?;
        if count as u64 > work.options.topology.maximum_tetrahedra {
            return Err(super::super::resource(
                "legalization cell inventory limit exceeded",
            ));
        }
        record_flip(work)?;
        for cell in [query.cell, right_index] {
            let tetrahedron = topology.tetrahedron(cell).expect("live removed cell");
            for opposite in 0..4 {
                pending.remove(&FaceQuery {
                    cell_key: key(tetrahedron.vertex_indices),
                    opposite,
                    cell,
                });
            }
        }
        let changed = topology
            .replace_cavity(&BTreeSet::from([query.cell, right_index]), created.to_vec())
            .map_err(super::super::mutation_topology_error)?;
        for cell in changed {
            enqueue(&topology, cell, focus, &mut pending);
            for neighbor in topology
                .tetrahedron(cell)
                .expect("committed cell")
                .neighbors
                .into_iter()
                .flatten()
            {
                enqueue(&topology, neighbor, focus, &mut pending);
            }
        }
    }
    topology
        .freeze(work.options.topology, work.cancellation())
        .map_err(super::super::mutation_topology_error)
}

fn candidate(
    topology: &TopologyMutation,
    query: FaceQuery,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    focus: Option<u32>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<(u32, [u32; 3])>, DelaunayInsertionError> {
    let Some(left) = topology.tetrahedron(query.cell) else {
        return Ok(None);
    };
    if key(left.vertex_indices) != query.cell_key {
        return Ok(None);
    }
    let Some(right_index) = left.neighbors[query.opposite] else {
        return Ok(None);
    };
    let right = topology
        .tetrahedron(right_index)
        .expect("live face neighbor");
    if focus.is_none() && key(right.vertex_indices) <= query.cell_key {
        return Ok(None);
    }
    work.checkpoint()?;
    if protected_faces.contains(&stable_face(topology, left.vertex_indices, query.opposite))
        || protect_region_boundaries && left.region_id != right.region_id
    {
        return Ok(None);
    }
    let right_apex = right
        .vertex_indices
        .into_iter()
        .find(|vertex| !left.vertex_indices.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "legalization neighbors do not share exactly one triangular face",
            )
        })?;
    let left_apex = left
        .vertex_indices
        .into_iter()
        .find(|vertex| !right.vertex_indices.contains(vertex))
        .ok_or_else(|| {
            error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "legalization neighbors do not have distinct opposite vertices",
            )
        })?;
    if circumsphere_relation(
        topology,
        left.vertex_indices,
        topology.nodes()[right_apex as usize],
        work,
    )? != PredicateSign::Positive
        || !two_to_three_topology_is_admissible(
            topology,
            query.cell,
            right_index,
            left_apex,
            right_apex,
        )
        || !two_to_three_geometry_is_admissible(
            left.vertex_indices,
            right.vertex_indices,
            |vertex| topology.nodes()[vertex as usize].materialized_predicate_point(),
        )
        .map_err(super::predicate_error)?
    {
        return Ok(None);
    }
    let mut face = [0; 3];
    let mut cursor = 0;
    for vertex in left.vertex_indices {
        if right.vertex_indices.contains(&vertex) {
            face[cursor] = vertex;
            cursor += 1;
        }
    }
    face.sort_unstable();
    Ok(Some((right_index, face)))
}

#[cfg(test)]
mod tests;
