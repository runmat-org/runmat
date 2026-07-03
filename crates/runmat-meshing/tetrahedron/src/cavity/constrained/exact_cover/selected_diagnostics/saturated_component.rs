use super::*;

pub fn selected_exact_cover_saturated_component(
    cavity: &ConstrainedCavity,
    selected_tetrahedron_node_ids: &[[u32; 4]],
    seed_face: [u32; 3],
) -> BoundaryExactCoverSaturatedComponent {
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let seed_face = sorted_face(seed_face);
    let selected_tetrahedra_by_face = selected_tetrahedra_by_face(selected_tetrahedron_node_ids);
    let saturated_faces = saturated_interior_faces(&boundary_faces, &selected_tetrahedra_by_face);
    let (component_faces, component_tetrahedra) =
        saturated_component_faces(seed_face, &saturated_faces, &selected_tetrahedra_by_face);

    BoundaryExactCoverSaturatedComponent {
        seed_face,
        saturated_face_count: saturated_faces.len(),
        component_face_count: component_faces.len(),
        component_tetrahedron_count: component_tetrahedra.len(),
        component_faces: component_faces.into_iter().collect(),
        component_tetrahedra: component_tetrahedra.into_iter().collect(),
    }
}

fn selected_tetrahedra_by_face(
    selected_tetrahedron_node_ids: &[[u32; 4]],
) -> BTreeMap<[u32; 3], Vec<[u32; 4]>> {
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }
    selected_tetrahedra_by_face
}

fn saturated_interior_faces(
    boundary_faces: &BTreeSet<[u32; 3]>,
    selected_tetrahedra_by_face: &BTreeMap<[u32; 3], Vec<[u32; 4]>>,
) -> BTreeSet<[u32; 3]> {
    selected_tetrahedra_by_face
        .iter()
        .filter_map(|(face, selected_tetrahedra)| {
            (!boundary_faces.contains(face) && selected_tetrahedra.len() >= 2).then_some(*face)
        })
        .collect()
}

fn saturated_component_faces(
    seed_face: [u32; 3],
    saturated_faces: &BTreeSet<[u32; 3]>,
    selected_tetrahedra_by_face: &BTreeMap<[u32; 3], Vec<[u32; 4]>>,
) -> (BTreeSet<[u32; 3]>, BTreeSet<[u32; 4]>) {
    let mut component_faces = BTreeSet::<[u32; 3]>::new();
    let mut component_tetrahedra = BTreeSet::<[u32; 4]>::new();
    let mut pending = Vec::<[u32; 3]>::new();
    if saturated_faces.contains(&seed_face) {
        pending.push(seed_face);
    }
    while let Some(face) = pending.pop() {
        if !component_faces.insert(face) {
            continue;
        }
        let Some(selected_tetrahedra) = selected_tetrahedra_by_face.get(&face) else {
            continue;
        };
        for selected_tetrahedron in selected_tetrahedra {
            if component_tetrahedra.insert(*selected_tetrahedron) {
                for adjacent_face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
                    if saturated_faces.contains(&adjacent_face)
                        && !component_faces.contains(&adjacent_face)
                    {
                        pending.push(adjacent_face);
                    }
                }
            }
        }
    }
    (component_faces, component_tetrahedra)
}
