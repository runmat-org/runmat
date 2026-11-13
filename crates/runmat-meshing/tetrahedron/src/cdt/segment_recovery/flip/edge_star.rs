//! Endpoint-linked edge-star reconnections and bounded local rotations.
use super::*;

pub(super) fn try_recover_edge_with_edge_star_rotation(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let edge_stars =
        candidate_edge_stars_linking_endpoints(topology, left, right, constraint_index, work)?;
    for (removed_edge, star) in edge_stars {
        if star.len() < 3
            || removed_edge.contains(&left)
            || removed_edge.contains(&right)
            || protected_edges.contains(&removed_edge)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| {
                topology
                    .tetrahedron(*index)
                    .expect("edge-star tetrahedron is active")
                    .vertex_indices
            })
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let candidates = crate::reconnect::BoundaryEdgeStarCandidates::new(&boundary, &mut || {
            work.search_flip_step(constraint_index)
        })?;
        let created = candidates.tetrahedra([left, right], &mut || {
            work.search_flip_step(constraint_index)
        })?;
        let boundary_matches = local_tetrahedron_boundary_faces(&created) == boundary;
        if created.len() < 2 || !boundary_matches {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let region = topology
            .tetrahedron(star[0])
            .expect("edge-star tetrahedron is active")
            .region_id
            .clone();
        let regions_match = !star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        });
        let crosses_protected =
            removed_internal_face_is_protected(topology, &star, protected_faces);
        if !regions_match || crosses_protected {
            continue;
        }
        let mut trial = topology.cavity_trial();
        let removed = star
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        match trial.replace_cavity(
            &removed,
            created
                .into_iter()
                .map(|tetrahedron| (tetrahedron, region.clone()))
                .collect(),
        ) {
            Ok(_) => {}
            Err(_) => continue,
        }
        trial.commit();
        return Ok(true);
    }
    Ok(false)
}

pub(super) fn try_recover_edge_with_three_to_two_mutation(
    topology: &mut TopologyMutation,
    left: u32,
    right: u32,
    protected_edges: &BTreeSet<[u32; 2]>,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    let edge_stars =
        candidate_edge_stars_linking_endpoints(topology, left, right, constraint_index, work)?;
    for (removed_edge, star) in edge_stars {
        if star.len() != 3
            || removed_edge.contains(&left)
            || removed_edge.contains(&right)
            || protected_edges.contains(&removed_edge)
        {
            continue;
        }
        work.flip_attempt(constraint_index)?;
        let local = std::array::from_fn(|index| {
            let tetrahedron = topology
                .tetrahedron(star[index])
                .expect("edge-star tetrahedron remains active");
            LocalTetrahedron {
                tetrahedron_id: star[index],
                node_ids: tetrahedron.vertex_indices,
            }
        });
        let Ok(candidate) = three_to_two_edge_flip_candidate(local, removed_edge) else {
            continue;
        };
        if !candidate
            .created_tetrahedra
            .iter()
            .any(|tetrahedron| tetrahedron.contains(&left) && tetrahedron.contains(&right))
        {
            continue;
        }
        let region = topology
            .tetrahedron(star[0])
            .expect("edge-star tetrahedron remains active")
            .region_id
            .clone();
        if star.iter().any(|index| {
            topology
                .tetrahedron(*index)
                .is_none_or(|tetrahedron| tetrahedron.region_id != region)
        }) || removed_internal_face_is_protected(topology, &star, protected_faces)
        {
            continue;
        }
        let mut trial = topology.cavity_trial();
        let removed = star
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        match trial.replace_cavity(
            &removed,
            candidate
                .created_tetrahedra
                .iter()
                .copied()
                .map(|tetrahedron| (tetrahedron, region.clone()))
                .collect(),
        ) {
            Ok(_) => {}
            Err(_) => continue,
        }
        let left_identity = trial.nodes()[left as usize].identity;
        let right_identity = trial.nodes()[right as usize].identity;
        if edge_exists(
            &*trial,
            left_identity,
            right_identity,
            constraint_index,
            work,
        )? {
            trial.commit();
            return Ok(true);
        }
    }
    Ok(false)
}

fn candidate_edge_stars_linking_endpoints(
    topology: &TopologyMutation,
    left: u32,
    right: u32,
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<std::collections::BTreeMap<[u32; 2], Vec<u32>>, DelaunaySegmentRecoveryError> {
    let incident_edges = |endpoint: u32,
                          work: &mut RecoveryWork<'_>|
     -> Result<BTreeSet<[u32; 2]>, DelaunaySegmentRecoveryError> {
        let mut edges = BTreeSet::new();
        for tetrahedron_index in topology.vertex_star(endpoint) {
            work.search_flip_step(constraint_index)?;
            let tetrahedron = topology.tetrahedron(tetrahedron_index).ok_or_else(|| {
                invalid_topology(
                    constraint_index,
                    "vertex-star incidence references an inactive tetrahedron",
                )
            })?;
            for first in 0..4 {
                for second in (first + 1)..4 {
                    let mut edge = [
                        tetrahedron.vertex_indices[first],
                        tetrahedron.vertex_indices[second],
                    ];
                    edge.sort_unstable();
                    if !edge.contains(&left) && !edge.contains(&right) {
                        edges.insert(edge);
                    }
                }
            }
        }
        Ok(edges)
    };
    let left_edges = incident_edges(left, work)?;
    let right_edges = incident_edges(right, work)?;
    let mut stars = std::collections::BTreeMap::new();
    for edge in left_edges.intersection(&right_edges).copied() {
        let mut star = Vec::new();
        for tetrahedron_index in topology.vertex_star(edge[0]) {
            work.search_flip_step(constraint_index)?;
            if topology
                .tetrahedron(tetrahedron_index)
                .is_some_and(|tetrahedron| tetrahedron.vertex_indices.contains(&edge[1]))
            {
                star.push(tetrahedron_index);
            }
        }
        star.sort_unstable();
        stars.insert(edge, star);
    }
    Ok(stars)
}
