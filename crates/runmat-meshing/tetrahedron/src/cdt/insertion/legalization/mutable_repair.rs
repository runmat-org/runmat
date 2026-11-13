//! Physical local repair over maintained incidence. Seed violations once, then
//! revisit only faces changed by atomic cavity replacement. Canonical output is
//! frozen once at the end; the caller independently validates the whole result.
use super::*;
use crate::cdt::{
    insertion::topology_access::TetrahedralTopology,
    mutation::{TopologyMutation, TopologyMutationErrorKind},
};
mod advanced;
mod candidates;

pub(in crate::cdt::insertion) fn restore(
    topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut topology = TopologyMutation::from_canonical(topology);
    let mut pending = BTreeSet::new();
    for index in 0..topology.tetrahedron_slot_count() {
        work.checkpoint()?;
        let Some(tetrahedron) = topology.tetrahedron(index as u32) else {
            continue;
        };
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some_and(|neighbor| neighbor > index as u32) {
                let face = face(tetrahedron.vertex_indices, opposite);
                if violation(&topology, face, protected_faces, work)?.is_some() {
                    pending.insert(face);
                }
            }
        }
    }
    restore_faces(&mut topology, pending, protected_faces, work)?;
    topology
        .freeze(work.options.topology, work.cancellation())
        .map_err(super::super::mutation_topology_error)
}

/// Repair only changes to a construction-verified unconstrained physical prefix.
/// Rejected work retains the complete dirty inventory; only an exact final
/// recheck can discharge it. Unrepairable changes take the ordinary full path.
pub(in crate::cdt) fn restore_admitted(
    topology: &mut TopologyMutation,
    work: &mut InsertionWork<'_>,
) -> Result<bool, DelaunayInsertionError> {
    if topology.physical_dirty_faces().is_none() {
        return Ok(false);
    }
    let barriers = BTreeSet::new();
    let mut visited = BTreeSet::new();
    loop {
        let mut pending = BTreeSet::new();
        for shared in topology
            .physical_dirty_faces()
            .expect("retained physical prefix")
        {
            work.checkpoint()?;
            pending.insert(*shared);
        }
        restore_faces(topology, pending, &barriers, work)?;
        let (before, queries) = advanced::violations(topology, work)?;
        if before == 0 {
            work.checkpoint()?;
            topology.discharge_physical_dirty_faces();
            return Ok(true);
        }
        let fingerprint = topology.connectivity_search_fingerprint(&mut || work.checkpoint())?;
        if !visited.insert(fingerprint)
            || !advanced::repair(topology, before, &queries, &visited, work)?
        {
            return Ok(false);
        }
    }
}

fn restore_faces(
    topology: &mut TopologyMutation,
    mut pending: BTreeSet<[u32; 3]>,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    while let Some(shared) = pending.pop_first() {
        work.checkpoint()?;
        let Some(sides) = violation(topology, shared, protected_faces, work)? else {
            continue;
        };
        let Some((removed, created)) =
            candidates::replacement(topology, shared, sides, protected_faces, work)?
        else {
            continue;
        };
        let region = topology
            .tetrahedron(sides[0].0)
            .expect("live face owner")
            .region_id
            .clone();
        let resulting_count = topology
            .active_tetrahedron_count()
            .checked_sub(removed.len())
            .and_then(|count| count.checked_add(created.len()))
            .ok_or_else(|| super::super::resource("mutable repair cell count overflow"))?;
        if resulting_count as u64 > work.options.topology.maximum_tetrahedra {
            return Err(super::super::resource(
                "mutable repair cell inventory limit exceeded",
            ));
        }
        record_flip(work)?;
        let created = match topology.replace_cavity(
            &removed,
            created
                .into_iter()
                .map(|vertices| (vertices, region.clone()))
                .collect(),
        ) {
            Ok(created) => created,
            Err(failure) if failure.kind == TopologyMutationErrorKind::InvalidTopology => continue,
            Err(failure) => return Err(super::super::mutation_topology_error(failure)),
        };
        for index in created {
            let tetrahedron = topology.tetrahedron(index).expect("committed repair cell");
            for opposite in 0..4 {
                let face = face(tetrahedron.vertex_indices, opposite);
                if violation(topology, face, protected_faces, work)?.is_some() {
                    pending.insert(face);
                }
            }
        }
    }
    Ok(())
}

fn violation(
    topology: &TopologyMutation,
    face: [u32; 3],
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<[(u32, usize); 2]>, DelaunayInsertionError> {
    let Some(sides) = topology.face_adjacency(face) else {
        return Ok(None);
    };
    if protected_faces.contains(&stable(topology, face)) {
        return Ok(None);
    }
    let left = topology.tetrahedron(sides[0].0).expect("live face owner");
    let right = topology.tetrahedron(sides[1].0).expect("live face owner");
    if left.region_id != right.region_id {
        return Ok(None);
    }
    Ok((sphere(
        topology,
        left.vertex_indices,
        right.vertex_indices[sides[1].1],
        work,
    )? == PredicateSign::Positive)
        .then_some(sides))
}

fn sphere(
    topology: &TopologyMutation,
    vertices: [u32; 4],
    query: u32,
    work: &mut InsertionWork<'_>,
) -> Result<PredicateSign, DelaunayInsertionError> {
    work.predicate()?;
    let mut points = vertices.map(|vertex| topology.nodes()[vertex as usize]);
    match topology
        .orient_nodes(points)
        .map_err(super::predicate_error)?
    {
        PredicateSign::Negative => points.swap(0, 1),
        PredicateSign::Zero => {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "physical repair encountered a zero-volume cell",
            ))
        }
        PredicateSign::Positive => {}
    }
    work.predicate()?;
    topology
        .insphere_nodes([
            points[0],
            points[1],
            points[2],
            points[3],
            topology.nodes()[query as usize],
        ])
        .map_err(super::predicate_error)
}

fn face(vertices: [u32; 4], opposite: usize) -> [u32; 3] {
    let mut result = [0; 3];
    let mut cursor = 0;
    for (slot, vertex) in vertices.into_iter().enumerate() {
        if slot != opposite {
            result[cursor] = vertex;
            cursor += 1;
        }
    }
    result.sort_unstable();
    result
}

fn stable(topology: &TopologyMutation, face: [u32; 3]) -> [StableDigest; 3] {
    let mut result = face.map(|vertex| topology.nodes()[vertex as usize].identity);
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests;
