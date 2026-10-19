//! Strictly improving physical flips with unchanged embedded cavity coverage.
use super::*;
use crate::cdt::insertion::cospherical::{boundary_diagonal_geometry, interior_diagonal_geometry};
type Replacement = (BTreeSet<u32>, Vec<[u32; 4]>);

pub(super) fn replacement(
    topology: &TopologyMutation,
    shared: [u32; 3],
    sides: [(u32, usize); 2],
    barriers: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<Replacement>, DelaunayInsertionError> {
    let left = topology.tetrahedron(sides[0].0).expect("live face owner");
    let right = topology.tetrahedron(sides[1].0).expect("live face owner");
    let endpoints = [
        left.vertex_indices[sides[0].1],
        right.vertex_indices[sides[1].1],
    ];
    work.predicate()?;
    if interior_diagonal_geometry(topology, endpoints, shared)
        .map_err(super::super::predicate_error)?
    {
        return Ok(Some((
            BTreeSet::from([sides[0].0, sides[1].0]),
            vec![
                [endpoints[0], endpoints[1], shared[0], shared[1]],
                [endpoints[0], endpoints[1], shared[1], shared[2]],
                [endpoints[0], endpoints[1], shared[2], shared[0]],
            ],
        )));
    }
    for edge in [
        [shared[0], shared[1]],
        [shared[1], shared[2]],
        [shared[0], shared[2]],
    ] {
        let apex = shared
            .into_iter()
            .find(|vertex| !edge.contains(vertex))
            .expect("face apex");
        let hull_faces = endpoints.map(|endpoint| {
            let mut face = [edge[0], edge[1], endpoint];
            face.sort_unstable();
            face
        });
        if hull_faces.iter().all(|face| {
            topology.boundary_face_owner(*face).is_some()
                && !barriers.contains(&stable(topology, *face))
        }) {
            work.predicate()?;
            if boundary_diagonal_geometry(topology, endpoints, edge, apex)
                .map_err(super::super::predicate_error)?
            {
                return Ok(Some((
                    BTreeSet::from([sides[0].0, sides[1].0]),
                    vec![
                        [endpoints[0], endpoints[1], edge[0], apex],
                        [endpoints[0], endpoints[1], apex, edge[1]],
                    ],
                )));
            }
        }
        if let Some(replacement) = reduce_edge(topology, edge, barriers, work)? {
            return Ok(Some(replacement));
        }
    }
    Ok(None)
}

fn reduce_edge(
    topology: &TopologyMutation,
    edge: [u32; 2],
    barriers: &BTreeSet<[StableDigest; 3]>,
    work: &mut InsertionWork<'_>,
) -> Result<Option<Replacement>, DelaunayInsertionError> {
    let mut star = Vec::new();
    for index in topology.vertex_star(edge[0]) {
        work.checkpoint()?;
        if topology
            .tetrahedron(index)
            .expect("live star")
            .vertex_indices
            .contains(&edge[1])
        {
            star.push(index);
            if star.len() > 3 {
                return Ok(None);
            }
        }
    }
    if star.len() != 3 {
        return Ok(None);
    }
    let region = &topology.tetrahedron(star[0]).expect("live star").region_id;
    let removed = star.iter().copied().collect::<BTreeSet<_>>();
    for index in &star {
        let tetrahedron = topology.tetrahedron(*index).expect("live star");
        if &tetrahedron.region_id != region {
            return Ok(None);
        }
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some_and(|neighbor| removed.contains(&neighbor))
                && barriers.contains(&stable(
                    topology,
                    face(tetrahedron.vertex_indices, opposite),
                ))
            {
                return Ok(None);
            }
        }
    }
    let local = std::array::from_fn(|slot| LocalTetrahedron {
        tetrahedron_id: star[slot],
        node_ids: topology
            .tetrahedron(star[slot])
            .expect("live star")
            .vertex_indices,
    });
    let Ok(candidate) = three_to_two_edge_flip_candidate(local, edge) else {
        return Ok(None);
    };
    let created = candidate.created_tetrahedra;
    work.predicate()?;
    if !two_to_three_geometry_is_admissible(created[0], created[1], |vertex| {
        topology.nodes()[vertex as usize].materialized_predicate_point()
    })
    .map_err(super::super::predicate_error)?
    {
        return Ok(None);
    }
    let query = created[1]
        .into_iter()
        .find(|vertex| !created[0].contains(vertex))
        .expect("new apex");
    // This repair loop follows strict physical improvement. A zero-energy
    // adjacent reduction can leave the triggering violation unchanged and be
    // reversed indefinitely. Cospherical reconnection has a separate provider.
    if sphere(topology, created[0], query, work)? != PredicateSign::Negative {
        return Ok(None);
    }
    Ok(Some((removed, created)))
}

#[cfg(test)]
mod tests;
