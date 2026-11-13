//! Exact crossing diagnostics over one caller-owned conservative cell inventory.
//! The inventory is reused only across read-only assessments of the same owner;
//! geometry, scoring and acceptance are unchanged, and mutations rederive it.
use super::*;

pub(super) fn crossing_face_obstruction_count(
    topology: &impl TetrahedralTopology,
    cells: &[u32],
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<usize, DelaunaySegmentRecoveryError> {
    let mut count = 0;
    for tetrahedron_index in cells.iter().copied() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor_index) = *neighbor else {
                continue;
            };
            if neighbor_index <= tetrahedron_index {
                continue;
            }
            work.search_flip_step(constraint_index)?;
            let face = face_opposite(tetrahedron.vertex_indices, opposite);
            if !segment_strictly_crosses_face(topology, [left, right], face, constraint_index)? {
                continue;
            }
            let neighbor = topology
                .tetrahedron(neighbor_index)
                .expect("crossing neighbor is active");
            let left_apex = tetrahedron.vertex_indices[opposite];
            let Some(right_apex) = neighbor
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !face.contains(vertex))
            else {
                count += 1;
                continue;
            };
            if !two_to_three_topology_is_admissible(
                topology,
                tetrahedron_index,
                neighbor_index,
                left_apex,
                right_apex,
            ) || !convex_flip_cavity(topology, left_apex, right_apex, face, constraint_index)?
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

pub(super) fn first_edge_crossing_face(
    topology: &impl TetrahedralTopology,
    cells: &[u32],
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<[u32; 3]>, DelaunaySegmentRecoveryError> {
    let mut candidates = std::collections::BTreeSet::new();
    for tetrahedron_index in cells.iter().copied() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_none_or(|neighbor| neighbor <= tetrahedron_index) {
                continue;
            }
            work.search_flip_step(constraint_index)?;
            let face = face_opposite(tetrahedron.vertex_indices, opposite);
            if segment_crosses_face_edge(topology, [left, right], face, constraint_index)? {
                candidates.insert(face);
            }
        }
    }
    Ok(candidates.into_iter().next())
}

fn segment_crosses_face_edge(
    topology: &impl TetrahedralTopology,
    segment: [u32; 2],
    face: [u32; 3],
    constraint_index: u32,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    if face.contains(&segment[0]) || face.contains(&segment[1]) {
        return Ok(false);
    }
    let orient = |indices: [u32; 4]| {
        orient_volume_nodes(indices.map(|index| topology.nodes()[index as usize])).map_err(
            |failure| {
                invalid_topology(
                    constraint_index,
                    format!("crossing-edge predicate failed: {failure:?}"),
                )
            },
        )
    };
    let side = [
        orient([face[0], face[1], face[2], segment[0]])?,
        orient([face[0], face[1], face[2], segment[1]])?,
    ];
    if side.contains(&PredicateSign::Zero) || side[0] == side[1] {
        return Ok(false);
    }
    let around = [
        orient([segment[0], segment[1], face[0], face[1]])?,
        orient([segment[0], segment[1], face[1], face[2]])?,
        orient([segment[0], segment[1], face[2], face[0]])?,
    ];
    let nonzero = around
        .iter()
        .copied()
        .filter(|sign| *sign != PredicateSign::Zero)
        .collect::<Vec<_>>();
    Ok(around
        .iter()
        .filter(|sign| **sign == PredicateSign::Zero)
        .count()
        == 1
        && nonzero.len() == 2
        && nonzero[0] == nonzero[1])
}

pub(super) fn crossing_face_count(
    topology: &impl TetrahedralTopology,
    cells: &[u32],
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<usize, DelaunaySegmentRecoveryError> {
    let mut count = 0;
    for tetrahedron_index in cells.iter().copied() {
        let Some(tetrahedron) = topology.tetrahedron(tetrahedron_index) else {
            continue;
        };
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            if neighbor.is_none_or(|neighbor| neighbor <= tetrahedron_index) {
                continue;
            }
            work.search_flip_step(constraint_index)?;
            if segment_strictly_crosses_face(
                topology,
                [left, right],
                face_opposite(tetrahedron.vertex_indices, opposite),
                constraint_index,
            )? {
                count += 1;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests;
