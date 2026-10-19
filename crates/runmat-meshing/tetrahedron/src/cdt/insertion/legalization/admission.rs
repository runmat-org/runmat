//! Shared embedded-cavity and materialized-volume admission.
use super::*;

pub(super) fn prospective_tetrahedra_form_embedded_manifold(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    removed: &[usize],
    created: &[[u32; 4]],
) -> Result<bool, DelaunayInsertionError> {
    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    // Legalization changes connectivity, never the admitted point inventory.
    // An expanded cavity must not strand an existing interior vertex, even
    // when its replacement has the same boundary and materialized volume.
    let retained_vertices = created.iter().flatten().copied().collect::<BTreeSet<_>>();
    for vertex in removed.iter().flat_map(|index| {
        topology
            .tetrahedron(*index as u32)
            .expect("live cavity")
            .vertex_indices
    }) {
        if !retained_vertices.contains(&vertex)
            && topology
                .vertex_star(vertex)
                .iter()
                .all(|index| removed.contains(&(*index as usize)))
        {
            return Ok(false);
        }
    }
    let mut created_keys = BTreeSet::new();
    let mut face_uses = BTreeMap::<[u32; 3], Vec<u32>>::new();
    for tetrahedron in created.iter().copied() {
        let mut key = tetrahedron;
        key.sort_unstable();
        if !created_keys.insert(key)
            || topology
                .vertex_star(key[0])
                .iter()
                .copied()
                .map(|index| index as usize)
                .filter(|index| !removed.contains(index))
                .any(|index| {
                    let mut retained = topology
                        .tetrahedron(index as u32)
                        .expect("live cavity")
                        .vertex_indices;
                    retained.sort_unstable();
                    retained == key
                })
        {
            return Ok(false);
        }
        for opposite in 0..4 {
            let mut face = [0u32; 3];
            let mut cursor = 0;
            for (slot, vertex) in tetrahedron.iter().enumerate() {
                if slot != opposite {
                    face[cursor] = *vertex;
                    cursor += 1;
                }
            }
            face.sort_unstable();
            let uses = face_uses.entry(face).or_default();
            uses.push(tetrahedron[opposite]);
            if uses.len() > 2 {
                return Ok(false);
            }
        }
    }
    for (face, mut apices) in face_uses {
        for index in topology
            .vertex_star(face[0])
            .iter()
            .copied()
            .map(|index| index as usize)
            .filter(|index| !removed.contains(index))
        {
            let tetrahedron = &topology.tetrahedron(index as u32).expect("live cavity");
            if tetrahedron.vertex_indices.contains(&face[1])
                && tetrahedron.vertex_indices.contains(&face[2])
            {
                apices.push(
                    tetrahedron
                        .vertex_indices
                        .iter()
                        .copied()
                        .find(|vertex| !face.contains(vertex))
                        .expect("retained face use has one opposite vertex"),
                );
            }
        }
        let [left, right] = apices.as_slice() else {
            if apices.len() > 2 {
                return Ok(false);
            }
            continue;
        };
        let face_nodes = face.map(|node| topology.nodes()[node as usize]);
        let left_sign = orient_volume_nodes([
            face_nodes[0],
            face_nodes[1],
            face_nodes[2],
            topology.nodes()[*left as usize],
        ])
        .map_err(super::predicate_error)?;
        let right_sign = orient_volume_nodes([
            face_nodes[0],
            face_nodes[1],
            face_nodes[2],
            topology.nodes()[*right as usize],
        ])
        .map_err(super::predicate_error)?;
        if left_sign == PredicateSign::Zero
            || right_sign == PredicateSign::Zero
            || left_sign == right_sign
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn apply_reconnection(
    topology: DelaunayVolumeTopology,
    removed: &[usize],
    created: Vec<[u32; 4]>,
    work: &InsertionWork<'_>,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    let region = topology.tetrahedra[removed[0]].region_id.clone();
    let removed = removed.iter().copied().collect::<BTreeSet<_>>();
    let mut tetrahedra = topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(_, tetrahedron)| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
        .collect::<Vec<_>>();
    tetrahedra.extend(
        created
            .into_iter()
            .map(|tetrahedron| (tetrahedron, region.clone())),
    );
    build_delaunay_volume_topology_with_regions(
        topology.nodes,
        tetrahedra,
        work.options.topology,
        work.cancellation(),
    )
    .map_err(topology_error)
}

pub(super) fn tetrahedra_are_materializable(
    topology: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    tetrahedra: &[[u32; 4]],
) -> Result<bool, DelaunayInsertionError> {
    for tetrahedron in tetrahedra {
        let nodes = tetrahedron.map(|node| topology.nodes()[node as usize]);
        let constructed =
            crate::cdt::topology::orient_volume_nodes(nodes).map_err(super::predicate_error)?;
        let materialized =
            orient_materialized_volume_nodes(nodes).map_err(super::predicate_error)?;
        if constructed == PredicateSign::Zero || materialized == PredicateSign::Zero {
            return Ok(false);
        }
    }
    Ok(true)
}
