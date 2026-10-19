//! Select repair neighborhoods by exact violations across unprotected neighbor faces.
//! Checked replacements descend the shared exact parabolic lifting functional.
//! Global completeness search retains its separate all-point empty-sphere scoring.
use super::*;

pub(super) fn first_candidate(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited: &BTreeSet<[u8; 32]>,
    state: crate::cdt::topology::ConnectivitySearchState,
    work: &mut InsertionWork<'_>,
) -> Result<Option<search_state::RepairCandidate>, DelaunayInsertionError> {
    let (before, queries) = violations(topology, protected_faces, work)?;
    let mut plateau = None;
    for query in queries {
        let Some((removed, created)) =
            vertex_reinsertion_geometry(topology, query, protected_faces, work)?
        else {
            continue;
        };
        let fingerprint =
            search_state::replacement(state, topology, &removed, &created, work)?.fingerprint();
        if visited.contains(&fingerprint) {
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
                let (after, _) = violations(&candidate.0, protected_faces, work)?;
                if after <= before {
                    plateau = Some(candidate);
                }
            }
            _ => {}
        }
    }
    Ok(plateau)
}

pub(super) fn violations(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<(u64, BTreeSet<usize>), DelaunayInsertionError> {
    let mut count = 0;
    let mut queries = BTreeSet::new();
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for (slot, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor) = neighbor.map(|neighbor| neighbor as usize) else {
                continue;
            };
            if neighbor <= index {
                continue;
            }
            work.checkpoint()?;
            let other = &topology.tetrahedra[neighbor];
            if tetrahedron.region_id != other.region_id
                || protected_faces.contains(&stable_face(
                    topology,
                    tetrahedron.vertex_indices,
                    slot,
                ))
            {
                continue;
            }
            let query = other
                .vertex_indices
                .iter()
                .copied()
                .find(|vertex| !tetrahedron.vertex_indices.contains(vertex))
                .expect("checked neighbors have distinct opposite vertices");
            if repair_sphere_relation(
                topology,
                tetrahedron.vertex_indices,
                topology.nodes[query as usize],
                SphereRelationPolicy::PhysicalAdmission,
                work,
            )? == PredicateSign::Positive
            {
                count += 1;
                queries.insert(query as usize);
                queries.insert(tetrahedron.vertex_indices[slot] as usize);
            }
        }
    }
    Ok((count, queries))
}
