//! Neighbor-driven edge-star rotation, including four-to-four reconnections.
//! Geometry and transaction admission are shared with complete legalization.
use super::*;

pub(super) fn first_candidate(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited: &BTreeSet<[u8; 32]>,
    state: crate::cdt::topology::ConnectivitySearchState,
    work: &mut InsertionWork<'_>,
) -> Result<Option<search_state::RepairCandidate>, DelaunayInsertionError> {
    let (before, queries) = reinsertion::violations(topology, protected_faces, work)?;
    let mut edges = BTreeSet::new();
    for query in queries {
        for index in &topology.incidence.vertex_stars[query] {
            let vertices = topology.tetrahedra[*index as usize].vertex_indices;
            for left in 0..4 {
                for right in (left + 1)..4 {
                    let mut edge = [vertices[left], vertices[right]];
                    edge.sort_unstable();
                    edges.insert(edge);
                }
            }
        }
    }
    let mut plateau = None;
    for edge in edges {
        work.checkpoint()?;
        let removed = topology.incidence.vertex_stars[edge[0] as usize]
            .iter()
            .map(|index| *index as usize)
            .filter(|index| {
                topology.tetrahedra[*index]
                    .vertex_indices
                    .contains(&edge[1])
            })
            .collect::<Vec<_>>();
        if removed.len() < 3
            || removed_internal_face_is_protected(topology, &removed, protected_faces)
        {
            continue;
        }
        let region = &topology.tetrahedra[removed[0]].region_id;
        if removed
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let boundary = local_tetrahedron_boundary_faces(
            &removed
                .iter()
                .map(|index| topology.tetrahedra[*index].vertex_indices)
                .collect::<Vec<_>>(),
        );
        let candidates = crate::reconnect::BoundaryEdgeStarCandidates::new(&boundary, &mut || {
            work.checkpoint()
        })?;
        for target in candidates.targets() {
            work.checkpoint()?;
            if target == edge || topology_has_edge(topology, target) {
                continue;
            }
            let Some(created) = edge_star_replacement(topology, &candidates, target, work)? else {
                continue;
            };
            let fingerprint =
                search_state::replacement(state, topology, &removed, &created, work)?.fingerprint();
            if visited.contains(&fingerprint)
                || replacement_duplicates_retained_tetrahedron(topology, &removed, &created, work)?
                || !prospective_tetrahedra_form_embedded_manifold(topology, &removed, &created)?
                || !tetrahedra_are_materializable(topology, &created)?
            {
                continue;
            }
            match progress::lifting_change(topology, &removed, &created, work)? {
                PredicateSign::Negative => {
                    return Ok(Some(search_state::apply(
                        topology.clone(),
                        state,
                        &removed,
                        created,
                        work,
                    )?))
                }
                PredicateSign::Zero if plateau.is_none() => {
                    let candidate =
                        search_state::apply(topology.clone(), state, &removed, created, work)?;
                    let (after, _) = reinsertion::violations(&candidate.0, protected_faces, work)?;
                    if after <= before {
                        plateau = Some(candidate);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(plateau)
}
