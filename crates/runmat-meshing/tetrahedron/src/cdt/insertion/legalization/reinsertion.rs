//! Shared cavity geometry; global and local repair retain separate scoring.
use super::*;

pub(super) fn vertex_reinsertion_candidate(
    topology: &DelaunayVolumeTopology,
    query: usize,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<ScoredCavityReplacement>, DelaunayInsertionError> {
    let Some((removed, created)) =
        vertex_reinsertion_geometry(topology, query, protected_faces, work)?
    else {
        return Ok(None);
    };
    let removed_violations = indexed_tetrahedra_global_violation_count(topology, &removed, work)?;
    let created_violations = tetrahedra_global_violation_count(topology, &created, work)?;
    if created_violations > removed_violations {
        return Ok(None);
    }
    Ok(Some((
        removed,
        created,
        removed_violations,
        created_violations,
    )))
}

pub(super) fn vertex_reinsertion_geometry(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    query: usize,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut removed = topology
        .vertex_star(query as u32)
        .iter()
        .map(|index| *index as usize)
        .collect::<BTreeSet<_>>();
    if removed.is_empty() {
        return Ok(None);
    }
    let query_node = topology.nodes()[query];
    let mut queue = removed
        .iter()
        .flat_map(|index| {
            topology
                .tetrahedron(*index as u32)
                .expect("live cavity")
                .neighbors
        })
        .flatten()
        .map(|index| index as usize)
        .collect::<VecDeque<_>>();
    while let Some(index) = queue.pop_front() {
        if removed.contains(&index) {
            continue;
        }
        let tetrahedron = &topology.tetrahedron(index as u32).expect("live cavity");
        work.predicate()?;
        let points = tetrahedron
            .vertex_indices
            .map(|node| topology.nodes()[node as usize]);
        if matches!(
            insphere_volume_nodes([points[0], points[1], points[2], points[3], query_node])
                .map_err(super::predicate_error)?,
            PredicateSign::Positive | PredicateSign::Zero
        ) {
            removed.insert(index);
            queue.extend(
                tetrahedron
                    .neighbors
                    .iter()
                    .flatten()
                    .map(|index| *index as usize),
            );
        }
    }
    shrink_reinsertion_cavity(topology, &mut removed, query as u32, work)?;
    let removed = removed.into_iter().collect::<Vec<_>>();
    if removed_internal_face_is_protected(topology, &removed, protected_faces) {
        return Ok(None);
    }
    let region = topology
        .tetrahedron(removed[0] as u32)
        .expect("live cavity")
        .region_id;
    if removed.iter().any(|index| {
        topology
            .tetrahedron(*index as u32)
            .expect("live cavity")
            .region_id
            != region
    }) {
        return Ok(None);
    }
    let removed_tetrahedra = removed
        .iter()
        .map(|index| {
            topology
                .tetrahedron(*index as u32)
                .expect("live cavity")
                .vertex_indices
        })
        .collect::<Vec<_>>();
    let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
    let mut created = Vec::new();
    for face in boundary
        .iter()
        .filter(|face| !face.contains(&(query as u32)))
    {
        let tetrahedron = [face[0], face[1], face[2], query as u32];
        if orient_volume_nodes(tetrahedron.map(|node| topology.nodes()[node as usize]))
            .map_err(super::predicate_error)?
            != PredicateSign::Zero
        {
            created.push(tetrahedron);
        }
    }
    created.sort_unstable();
    created.dedup();
    let boundary_matches = local_tetrahedron_boundary_faces(&created) == boundary;
    let duplicates =
        replacement_duplicates_retained_tetrahedron(topology, &removed, &created, work)?;
    let embedded = prospective_tetrahedra_form_embedded_manifold(topology, &removed, &created)?;
    if created.is_empty()
        || !boundary_matches
        || duplicates
        || !embedded
        || !tetrahedra_are_materializable(topology, &created)?
    {
        return Ok(None);
    }
    Ok(Some((removed, created)))
}

pub(super) fn shrink_reinsertion_cavity(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    cavity: &mut BTreeSet<usize>,
    query: u32,
    work: &mut InsertionWork<'_>,
) -> Result<(), DelaunayInsertionError> {
    loop {
        let mut cut = None;
        for index in cavity.iter().copied() {
            let tetrahedron = &topology.tetrahedron(index as u32).expect("live cavity");
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| cavity.contains(&(neighbor as usize)))
                {
                    continue;
                }
                let mut face = [0; 3];
                let mut cursor = 0;
                for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
                    if slot != opposite {
                        face[cursor] = *vertex;
                        cursor += 1;
                    }
                }
                if face.contains(&query) {
                    continue;
                }
                let face_nodes = face.map(|node| topology.nodes()[node as usize]);
                let interior = topology.nodes()[tetrahedron.vertex_indices[opposite] as usize];
                work.predicate()?;
                let interior_sign =
                    orient_volume_nodes([face_nodes[0], face_nodes[1], face_nodes[2], interior])
                        .map_err(super::predicate_error)?;
                work.predicate()?;
                let query_sign = orient_volume_nodes([
                    face_nodes[0],
                    face_nodes[1],
                    face_nodes[2],
                    topology.nodes()[query as usize],
                ])
                .map_err(super::predicate_error)?;
                if query_sign == PredicateSign::Zero || query_sign != interior_sign {
                    cut = Some(index);
                    break;
                }
            }
            if cut.is_some() {
                break;
            }
        }
        if cut.is_none() {
            // Sphere expansion can swallow another vertex's complete star in
            // a non-Delaunay input. Coning that cavity only to `query` would
            // silently delete the other admitted point. Expose it on the
            // cavity boundary before performing the existing star-shape check.
            let boundary_vertices = local_tetrahedron_boundary_faces(
                &cavity
                    .iter()
                    .map(|index| {
                        topology
                            .tetrahedron(*index as u32)
                            .expect("live cavity")
                            .vertex_indices
                    })
                    .collect::<Vec<_>>(),
            )
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
            let vertices = cavity
                .iter()
                .flat_map(|index| {
                    topology
                        .tetrahedron(*index as u32)
                        .expect("live cavity")
                        .vertex_indices
                })
                .filter(|vertex| *vertex != query && !boundary_vertices.contains(vertex))
                .collect::<BTreeSet<_>>();
            for vertex in vertices {
                work.checkpoint()?;
                let star = &topology.vertex_star(vertex);
                if star.iter().all(|index| cavity.contains(&(*index as usize))) {
                    cut = star.iter().map(|index| *index as usize).find(|index| {
                        !topology
                            .tetrahedron(*index as u32)
                            .expect("live cavity")
                            .vertex_indices
                            .contains(&query)
                    });
                    if cut.is_none() {
                        return Err(error(
                            DelaunayInsertionErrorKind::InvalidTopology,
                            "vertex reinsertion cannot preserve an interior admitted point",
                        ));
                    }
                    break;
                }
            }
        }
        let Some(index) = cut else {
            return Ok(());
        };
        cavity.remove(&index);
        if cavity.is_empty() {
            return Err(error(
                DelaunayInsertionErrorKind::InvalidTopology,
                "global vertex reinsertion cavity validation removed every tetrahedron",
            ));
        }
    }
}

#[cfg(test)]
mod tests;
