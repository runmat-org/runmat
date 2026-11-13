//! Checked coplanar hull-diagonal reconnection.
use super::*;

pub(super) fn first_illegal_boundary_edge(
    topology: &DelaunayVolumeTopology,
    protected_faces: &BTreeSet<[StableDigest; 3]>,
    policy: SphereRelationPolicy,
    work: &mut InsertionWork<'_>,
) -> Result<Option<CavityReplacement>, DelaunayInsertionError> {
    let mut edge_uses = BTreeMap::<[u32; 2], Vec<(usize, [u32; 3], u32)>>::new();
    // Canonical incidence already owns the exact hull inventory. Visit that
    // inventory directly rather than rediscovering it through every volume cell.
    for facet in &topology.incidence.boundary_facets {
        work.checkpoint()?;
        let tetrahedron_index = facet.tetrahedron_index as usize;
        let tetrahedron = &topology.tetrahedra[tetrahedron_index];
        let face = facet.vertex_indices;
        for mut edge in [[face[0], face[1]], [face[1], face[2]], [face[0], face[2]]] {
            edge.sort_unstable();
            edge_uses.entry(edge).or_default().push((
                tetrahedron_index,
                face,
                tetrahedron.vertex_indices[facet.opposite_vertex_slot as usize],
            ));
        }
    }
    // Preserve the original ascending tetrahedron-side selection independently
    // of the canonical boundary-facet ordering.
    for uses in edge_uses.values_mut() {
        uses.sort_by_key(|entry| entry.0);
    }
    for (edge, uses) in edge_uses {
        let [(left_index, left_face, left_apex), (right_index, right_face, right_apex)] =
            uses.as_slice()
        else {
            continue;
        };
        work.checkpoint()?;
        if topology.tetrahedra[*left_index].region_id != topology.tetrahedra[*right_index].region_id
            || protected_faces.contains(&stable_face(
                topology,
                topology.tetrahedra[*left_index].vertex_indices,
                topology.tetrahedra[*left_index]
                    .vertex_indices
                    .iter()
                    .position(|vertex| *vertex == *left_apex)
                    .expect("boundary apex belongs to tetrahedron"),
            ))
            || protected_faces.contains(&stable_face(
                topology,
                topology.tetrahedra[*right_index].vertex_indices,
                topology.tetrahedra[*right_index]
                    .vertex_indices
                    .iter()
                    .position(|vertex| *vertex == *right_apex)
                    .expect("boundary apex belongs to tetrahedron"),
            ))
        {
            continue;
        }
        let left_boundary_apex = left_face
            .iter()
            .copied()
            .find(|vertex| !edge.contains(vertex))
            .expect("boundary face has one apex opposite its edge");
        let right_boundary_apex = right_face
            .iter()
            .copied()
            .find(|vertex| !edge.contains(vertex))
            .expect("boundary face has one apex opposite its edge");
        let coplanar = orient_volume_nodes([
            topology.nodes[edge[0] as usize],
            topology.nodes[edge[1] as usize],
            topology.nodes[left_boundary_apex as usize],
            topology.nodes[right_boundary_apex as usize],
        ])
        .map_err(super::predicate_error)?
            == PredicateSign::Zero;
        if !coplanar
            || repair_sphere_relation(
                topology,
                topology.tetrahedra[*left_index].vertex_indices,
                topology.nodes[right_boundary_apex as usize],
                policy,
                work,
            )? != PredicateSign::Positive
        {
            continue;
        }
        if left_apex == right_apex {
            let created = vec![
                [left_boundary_apex, right_boundary_apex, edge[0], *left_apex],
                [left_boundary_apex, right_boundary_apex, *left_apex, edge[1]],
            ];
            if prospective_tetrahedra_form_embedded_manifold(
                topology,
                &[*left_index, *right_index],
                &created,
            )? && tetrahedra_are_materializable(topology, &created)?
            {
                return Ok(Some((vec![*left_index, *right_index], created)));
            }
            continue;
        }
        let star = topology.incidence.vertex_stars[edge[0] as usize]
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| {
                topology.tetrahedra[*index]
                    .vertex_indices
                    .contains(&edge[1])
            })
            .collect::<Vec<_>>();
        if star.len() < 2 {
            continue;
        }
        let region = &topology.tetrahedra[star[0]].region_id;
        if star
            .iter()
            .any(|index| &topology.tetrahedra[*index].region_id != region)
        {
            continue;
        }
        let removed_tetrahedra = star
            .iter()
            .map(|index| topology.tetrahedra[*index].vertex_indices)
            .collect::<Vec<_>>();
        let boundary = local_tetrahedron_boundary_faces(&removed_tetrahedra);
        let mut removed_internal_faces = BTreeSet::new();
        for index in &star {
            let tetrahedron = &topology.tetrahedra[*index];
            for opposite in 0..4 {
                if tetrahedron.neighbors[opposite]
                    .is_some_and(|neighbor| star.contains(&(neighbor as usize)))
                {
                    removed_internal_faces.insert(stable_face(
                        topology,
                        tetrahedron.vertex_indices,
                        opposite,
                    ));
                }
            }
        }
        if removed_internal_faces
            .iter()
            .any(|face| protected_faces.contains(face))
        {
            continue;
        }
        let cavity_nodes = removed_tetrahedra
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let other = cavity_nodes
            .iter()
            .copied()
            .filter(|node| *node != left_boundary_apex && *node != right_boundary_apex)
            .collect::<Vec<_>>();
        let mut created = Vec::new();
        for first in 0..other.len() {
            for second in (first + 1)..other.len() {
                let mut left = [left_boundary_apex, other[first], other[second]];
                let mut right = [right_boundary_apex, other[first], other[second]];
                left.sort_unstable();
                right.sort_unstable();
                if boundary.contains(&left) && boundary.contains(&right) {
                    let tetrahedron = [
                        left_boundary_apex,
                        right_boundary_apex,
                        other[first],
                        other[second],
                    ];
                    if orient_volume_nodes(tetrahedron.map(|node| topology.nodes[node as usize]))
                        .map_err(super::predicate_error)?
                        != PredicateSign::Zero
                    {
                        created.push(tetrahedron);
                    }
                }
            }
        }
        if created.len() < 2 || local_tetrahedron_boundary_faces(&created) != boundary {
            continue;
        }
        if !prospective_tetrahedra_form_embedded_manifold(topology, &star, &created)?
            || !tetrahedra_are_materializable(topology, &created)?
        {
            continue;
        }
        return Ok(Some((star, created)));
    }
    Ok(None)
}
