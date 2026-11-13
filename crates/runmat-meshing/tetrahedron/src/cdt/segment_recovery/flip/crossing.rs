//! Crossing-face search retains bounded cycle history and exact geometry tests.
use super::*;
mod inventory;
mod reduce;
use inventory::{crossing_face_count, crossing_face_obstruction_count, first_edge_crossing_face};
use reduce::try_reduce_crossing_face_edge;

use super::spatial_candidates::candidate_cells;

pub(super) fn try_recover_edge_by_crossing_face_flips(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let mut visited = std::collections::BTreeSet::new();
    let constraints = CrossingFaceConstraints {
        target: [left, right],
        protected_edges,
        protected_faces,
        constraint_index,
    };
    search_crossing_face_flips(topology, &constraints, work, &mut visited)
}

struct CrossingFaceConstraints<'a> {
    target: [u32; 2],
    protected_edges: &'a BTreeSet<[u32; 2]>,
    protected_faces: &'a [[StableDigest; 3]],
    constraint_index: u32,
}

fn search_crossing_face_flips(
    topology: &mut TopologyMutation,
    constraints: &CrossingFaceConstraints<'_>,
    work: &mut RecoveryWork<'_>,
    visited: &mut std::collections::BTreeSet<[u8; 32]>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let [left, right] = constraints.target;
    let constraint_index = constraints.constraint_index;
    loop {
        let state = topology_state(topology, &mut || work.search_flip_step(constraint_index))?;
        if !visited.insert(state) {
            return Ok(false);
        }
        // Retain this exact broad-phase inventory only while the owner is
        // unchanged. Rejected cavity trials restore it; every accepted edit
        // returns to the loop and derives a fresh inventory.
        let cells = candidate_cells(topology, left, right, constraint_index, work)?;
        let initial_crossings =
            crossing_face_count(topology, &cells, left, right, constraint_index, work)?;
        if initial_crossings == 0 {
            if let Some(face) =
                first_edge_crossing_face(topology, &cells, left, right, constraint_index, work)?
            {
                if try_reduce_crossing_face_edge(topology, face, 0, constraints, work, visited)? {
                    continue;
                }
            }
            return Ok(false);
        }
        let mut accepted = None;
        'tetrahedra: for left_tetrahedron in cells {
            let Some(tetrahedron) = topology.tetrahedron(left_tetrahedron).cloned() else {
                continue;
            };
            for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
                let Some(right_tetrahedron) = *neighbor else {
                    continue;
                };
                if right_tetrahedron <= left_tetrahedron {
                    continue;
                }
                work.search_flip_step(constraint_index)?;
                let face = face_opposite(tetrahedron.vertex_indices, opposite);
                if !segment_strictly_crosses_face(topology, [left, right], face, constraint_index)?
                    || constraints
                        .protected_faces
                        .contains(&stable_face(topology, face))
                {
                    continue;
                }
                let neighbor = topology
                    .tetrahedron(right_tetrahedron)
                    .expect("crossing-face neighbor is active");
                if tetrahedron.region_id != neighbor.region_id {
                    continue;
                }
                let left_apex = tetrahedron.vertex_indices[opposite];
                let Some(right_apex) = neighbor
                    .vertex_indices
                    .iter()
                    .copied()
                    .find(|vertex| !face.contains(vertex))
                else {
                    continue;
                };
                work.flip_attempt(constraint_index)?;
                let topology_admissible = two_to_three_topology_is_admissible(
                    topology,
                    left_tetrahedron,
                    right_tetrahedron,
                    left_apex,
                    right_apex,
                );
                let geometry_admissible =
                    convex_flip_cavity(topology, left_apex, right_apex, face, constraint_index)?;
                if !topology_admissible {
                    continue;
                }
                if !geometry_admissible {
                    if try_reduce_crossing_face_edge(
                        topology,
                        face,
                        initial_crossings,
                        constraints,
                        work,
                        visited,
                    )? {
                        accepted = Some(false);
                        break 'tetrahedra;
                    }
                    continue;
                }
                let mut trial = topology.cavity_trial();
                let removed = [left_tetrahedron, right_tetrahedron]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>();
                let tetrahedron_region = tetrahedron.region_id.clone();
                let created = [
                    [left_apex, right_apex, face[0], face[1]],
                    [left_apex, right_apex, face[1], face[2]],
                    [left_apex, right_apex, face[2], face[0]],
                ]
                .into_iter()
                .map(|tetrahedron| (tetrahedron, tetrahedron_region.clone()))
                .collect();
                match trial.replace_cavity(&removed, created) {
                    Ok(_) => {}
                    Err(_) => continue,
                }
                let recovered = trial.vertex_star(left).into_iter().any(|index| {
                    trial
                        .tetrahedron(index)
                        .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&right))
                });
                let trial_cells = candidate_cells(&*trial, left, right, constraint_index, work)?;
                let remaining = crossing_face_count(
                    &*trial,
                    &trial_cells,
                    left,
                    right,
                    constraint_index,
                    work,
                )?;
                if !recovered && remaining >= initial_crossings {
                    continue;
                }
                trial.commit();
                accepted = Some(recovered);
                break 'tetrahedra;
            }
        }
        let Some(recovered) = accepted else {
            return Ok(false);
        };
        if recovered {
            return Ok(true);
        }
    }
}

fn segment_strictly_crosses_face(
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
                    format!("crossing-face predicate failed: {failure:?}"),
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
    Ok(around[0] != PredicateSign::Zero && around.iter().all(|sign| *sign == around[0]))
}
