//! Complete global repair coordination and original all-point progress rules.
use super::*;

pub(in crate::cdt::insertion) fn legalize_delaunay_volume_topology_complete(
    mut topology: DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    protect_region_boundaries: bool,
    work: &mut InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let mut state = search_state::seed(&topology, work)?;
    let mut visited_global_states = BTreeSet::new();
    let globally_unconstrained =
        protected_faces.is_empty() && topology.incidence.regions.is_empty();
    loop {
        // A repeated connectivity cannot provide new repair evidence. Return
        // the stalled attempt for independent admission, never as a certificate.
        if !visited_global_states.insert(state.fingerprint()) {
            return Ok(topology);
        }
        if let Some((removed, created)) = first_illegal_boundary_edge(
            &topology,
            protected_faces,
            SphereRelationPolicy::CanonicalInsertion,
            work,
        )? {
            record_flip(work)?;
            (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
            continue;
        }
        if globally_unconstrained {
            if let Some((removed, created)) = first_global_vertex_reinsertion(
                &topology,
                protected_faces,
                &visited_global_states,
                state,
                work,
            )? {
                record_flip(work)?;
                (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
                continue;
            }
            if let Some((removed, created)) = first_globally_delaunay_edge_rotation(
                &topology,
                protected_faces,
                &visited_global_states,
                state,
                work,
            )? {
                record_flip(work)?;
                (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
                continue;
            }
        }
        if let Some((removed, created)) = first_reducible_edge(
            &topology,
            protected_faces,
            protect_region_boundaries,
            false,
            work,
        )? {
            record_flip(work)?;
            (topology, state) = search_state::apply(topology, state, &removed, created, work)?;
            continue;
        }
        let Some((left, right, face)) = first_illegal_face(
            &topology,
            protected_faces,
            protect_region_boundaries,
            None,
            SphereRelationPolicy::CanonicalInsertion,
            work,
        )?
        else {
            return Ok(topology);
        };
        record_flip(work)?;
        let created = two_to_three::replacement(&topology, left, right, face)?;
        (topology, state) = search_state::apply(topology, state, &[left, right], created, work)?;
    }
}

fn first_global_vertex_reinsertion(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited_states: &BTreeSet<[u8; 32]>,
    state: crate::cdt::topology::ConnectivitySearchState,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut plateau_candidate = None;
    for query in global_delaunay_violation_queries(topology, work)? {
        let Some((removed, created, removed_violations, created_violations)) =
            vertex_reinsertion_candidate(topology, query, protected_faces, work)?
        else {
            continue;
        };
        if created_violations < removed_violations {
            return Ok(Some((removed, created)));
        }
        if created_violations == removed_violations
            && plateau_candidate.is_none()
            && !visited_states.contains(
                &search_state::replacement(state, topology, &removed, &created, work)?
                    .fingerprint(),
            )
        {
            plateau_candidate = Some((removed, created));
        }
    }
    Ok(plateau_candidate)
}

fn first_globally_delaunay_edge_rotation(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    visited_states: &BTreeSet<[u8; 32]>,
    state: crate::cdt::topology::ConnectivitySearchState,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let Some((violating_tetrahedron, query_node)) =
        first_global_delaunay_violation(topology, work)?
    else {
        return Ok(None);
    };
    let violating_vertices = topology.tetrahedra[violating_tetrahedron].vertex_indices;
    let mut candidate_edges = BTreeSet::new();
    for tetrahedron_index in std::iter::once(violating_tetrahedron).chain(
        topology.incidence.vertex_stars[query_node]
            .iter()
            .map(|index| *index as usize),
    ) {
        let vertices = topology.tetrahedra[tetrahedron_index].vertex_indices;
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [vertices[left], vertices[right]];
                edge.sort_unstable();
                candidate_edges.insert(edge);
            }
        }
    }
    let mut plateau_candidate = None;
    for removed_edge in candidate_edges {
        let star = topology.incidence.vertex_stars[removed_edge[0] as usize]
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| {
                topology.tetrahedra[*index]
                    .vertex_indices
                    .contains(&removed_edge[1])
            })
            .collect::<Vec<_>>();
        if star.len() < 3 {
            continue;
        }
        let removed_violations = indexed_tetrahedra_global_violation_count(topology, &star, work)?;
        if removed_violations == 0
            && !removed_edge
                .iter()
                .any(|vertex| violating_vertices.contains(vertex))
        {
            continue;
        }
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
            || removed_internal_face_is_protected(topology, &star, protected_faces)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let candidates = crate::reconnect::BoundaryEdgeStarCandidates::new(&boundary, &mut || {
            work.checkpoint()
        })?;
        for target in candidates.targets() {
            if target == removed_edge || topology_has_edge(topology, target) {
                continue;
            }
            let Some(created) = edge_star_replacement(topology, &candidates, target, work)? else {
                continue;
            };
            if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)?
                || replacement_duplicates_retained_tetrahedron(topology, &star, &created, work)?
                || !tetrahedra_are_materializable(topology, &created)?
            {
                continue;
            }
            let created_violations = tetrahedra_global_violation_count(topology, &created, work)?;
            if created_violations > removed_violations {
                continue;
            }
            if created_violations < removed_violations {
                return Ok(Some((star, created)));
            }
            if plateau_candidate.is_none()
                && target.contains(&(query_node as u32))
                && target
                    .iter()
                    .any(|vertex| violating_vertices.contains(vertex))
                && !visited_states.contains(
                    &search_state::replacement(state, topology, &star, &created, work)?
                        .fingerprint(),
                )
            {
                plateau_candidate = Some((star.clone(), created));
            }
        }
    }
    Ok(plateau_candidate)
}
