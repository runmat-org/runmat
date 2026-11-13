//! Checked three-to-two edge-star repair.
use super::*;

pub(super) fn first_reducible_edge(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    _protect_region_boundaries: bool,
    only_illegal_edges: bool,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut edge_stars = BTreeMap::<[u32; 2], Vec<usize>>::new();
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [
                    tetrahedron.vertex_indices[left],
                    tetrahedron.vertex_indices[right],
                ];
                edge.sort_unstable();
                edge_stars.entry(edge).or_default().push(tetrahedron_index);
            }
        }
    }
    for (edge, star) in edge_stars {
        if star.len() != 3 {
            continue;
        }
        work.checkpoint()?;
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let local = std::array::from_fn(|slot| LocalTetrahedron {
            tetrahedron_id: star[slot] as u32,
            node_ids: topology.tetrahedra[star[slot]].vertex_indices,
        });
        let Ok(candidate) = three_to_two_edge_flip_candidate(local, edge) else {
            continue;
        };
        if only_illegal_edges {
            let vertices = local
                .iter()
                .flat_map(|tetrahedron| tetrahedron.node_ids)
                .collect::<BTreeSet<_>>();
            let mut illegal = false;
            for tetrahedron in &local {
                for query in vertices
                    .iter()
                    .filter(|query| !tetrahedron.node_ids.contains(query))
                {
                    illegal |= repair_sphere_relation(
                        topology,
                        tetrahedron.node_ids,
                        topology.nodes[*query as usize],
                        SphereRelationPolicy::PhysicalAdmission,
                        work,
                    )? == PredicateSign::Positive;
                }
            }
            if !illegal {
                continue;
            }
        }
        let mut removed_faces = BTreeSet::new();
        for index in &star {
            let tetrahedron = &topology.tetrahedra[*index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| star.contains(&(neighbor as usize)))
                {
                    removed_faces.insert(stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ));
                }
            }
        }
        if removed_faces
            .iter()
            .any(|face| protected_faces.contains(face))
        {
            continue;
        }
        let mut created: [[u32; 4]; 2] = candidate
            .created_tetrahedra
            .try_into()
            .expect("three-to-two reconnection creates two tetrahedra");
        let mut nondegenerate = true;
        for tetrahedron in &mut created {
            match orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
                .map_err(super::predicate_error)?
            {
                PredicateSign::Positive => {}
                PredicateSign::Negative => tetrahedron.swap(0, 1),
                PredicateSign::Zero => nondegenerate = false,
            }
        }
        if !nondegenerate {
            continue;
        }
        let created_keys = created.map(|mut tetrahedron| {
            tetrahedron.sort_unstable();
            tetrahedron
        });
        if created_keys[0] == created_keys[1]
            || topology
                .tetrahedra
                .iter()
                .enumerate()
                .filter(|(index, _)| !star.contains(index))
                .any(|(_, tetrahedron)| {
                    let mut key = tetrahedron.vertex_indices;
                    key.sort_unstable();
                    created_keys.contains(&key)
                })
        {
            continue;
        }
        if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)? {
            continue;
        }
        if !tetrahedra_are_materializable(topology, &created)? {
            continue;
        }
        if !two_to_three_geometry_is_admissible(created[0], created[1], |node| {
            topology.nodes[node as usize].materialized_predicate_point()
        })
        .map_err(super::predicate_error)?
        {
            continue;
        }
        let opposite = created[1]
            .iter()
            .copied()
            .find(|node| !created[0].contains(node))
            .expect("inverse face flip has an opposite node");
        if repair_sphere_relation(
            topology,
            created[0],
            topology.nodes[opposite as usize],
            if only_illegal_edges {
                SphereRelationPolicy::PhysicalAdmission
            } else {
                SphereRelationPolicy::CanonicalInsertion
            },
            work,
        )? == PredicateSign::Positive
        {
            continue;
        }
        return Ok(Some((star, created.into_iter().collect())));
    }
    Ok(None)
}
