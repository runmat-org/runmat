//! Reduce obstructing edge stars while preserving the crossing-search criteria.
use super::*;
type ScoredReplacement = (
    usize,
    usize,
    BTreeSet<u32>,
    Vec<([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
);

pub(super) fn try_reduce_crossing_face_edge(
    topology: &mut TopologyMutation,
    face: [u32; 3],
    initial_crossings: usize,
    constraints: &CrossingFaceConstraints<'_>,
    work: &mut RecoveryWork<'_>,
    visited: &mut std::collections::BTreeSet<[u8; 32]>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let [target_left, target_right] = constraints.target;
    let constraint_index = constraints.constraint_index;
    let original_cells =
        candidate_cells(topology, target_left, target_right, constraint_index, work)?;
    let initial_obstructions = crossing_face_obstruction_count(
        topology,
        &original_cells,
        target_left,
        target_right,
        constraint_index,
        work,
    )?;
    let mut best = None::<ScoredReplacement>;
    for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
        edge.sort_unstable();
        if constraints.protected_edges.contains(&edge) {
            continue;
        }
        let star = topology
            .vertex_star(edge[0])
            .into_iter()
            .filter(|index| {
                topology
                    .tetrahedron(*index)
                    .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&edge[1]))
            })
            .collect::<Vec<_>>();
        if star.len() < 3 {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let removed_tetrahedra = star
            .iter()
            .map(|index| {
                topology
                    .tetrahedron(*index)
                    .expect("crossing edge-star tetrahedron is active")
                    .vertex_indices
            })
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let region = topology
            .tetrahedron(star[0])
            .expect("crossing edge-star tetrahedron is active")
            .region_id
            .clone();
        if star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        }) || removed_internal_face_is_protected(topology, &star, constraints.protected_faces)
        {
            continue;
        }
        let candidates = crate::reconnect::BoundaryEdgeStarCandidates::new(&boundary, &mut || {
            work.search_flip_step(constraint_index)
        })?;
        for diagonal in candidates.targets() {
            if diagonal == edge {
                continue;
            }
            let replacement =
                candidates.tetrahedra(diagonal, &mut || work.search_flip_step(constraint_index))?;
            if replacement.len() < 2 || local_tetrahedron_boundary_faces(&replacement) != boundary {
                continue;
            }
            let mut trial = topology.cavity_trial();
            let removed = star
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            match trial.replace_cavity(
                &removed,
                replacement
                    .iter()
                    .copied()
                    .map(|tetrahedron| (tetrahedron, region.clone()))
                    .collect(),
            ) {
                Ok(_) => {}
                Err(_) => continue,
            }
            let trial_cells =
                candidate_cells(&*trial, target_left, target_right, constraint_index, work)?;
            let remaining = crossing_face_count(
                &*trial,
                &trial_cells,
                target_left,
                target_right,
                constraint_index,
                work,
            )?;
            let obstructions = crossing_face_obstruction_count(
                &*trial,
                &trial_cells,
                target_left,
                target_right,
                constraint_index,
                work,
            )?;
            if (remaining, obstructions) > (initial_crossings, initial_obstructions) {
                continue;
            }
            if (remaining, obstructions) == (initial_crossings, initial_obstructions) {
                if initial_crossings == 0 {
                    trial.commit();
                    return Ok(true);
                }
                if search_crossing_face_flips(&mut trial, constraints, work, visited)? {
                    trial.commit();
                    return Ok(true);
                }
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|current| (remaining, obstructions) < (current.0, current.1))
            {
                best = Some((
                    remaining,
                    obstructions,
                    removed,
                    replacement
                        .iter()
                        .copied()
                        .map(|cell| (cell, region.clone()))
                        .collect::<Vec<_>>(),
                ));
            }
        }
    }
    if let Some((_, _, removed, created)) = best {
        topology
            .replace_cavity(&removed, created)
            .map_err(super::super::super::mutation_error)?;
        Ok(true)
    } else {
        Ok(false)
    }
}
