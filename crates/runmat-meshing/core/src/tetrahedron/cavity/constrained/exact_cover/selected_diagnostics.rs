use super::*;

pub fn selected_exact_cover_face_count_blockers(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCountBlockers, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let target_face = sorted_face(target_face);
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }

    let mut blockers = Vec::<BoundaryExactCoverFaceCountBlocker>::new();
    let mut candidate_count = 0_usize;
    for first in 0..node_ids.len() {
        for second in (first + 1)..node_ids.len() {
            for third in (second + 1)..node_ids.len() {
                for fourth in (third + 1)..node_ids.len() {
                    let tetrahedron_node_ids = [
                        node_ids[first],
                        node_ids[second],
                        node_ids[third],
                        node_ids[fourth],
                    ];
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .contains(&target_face)
                    {
                        continue;
                    }
                    if selected_keys.contains(&sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| boundary_node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) else {
                        continue;
                    };
                    candidate_count += 1;
                    let mut conflicting_faces = Vec::<[u32; 3]>::new();
                    let mut blocking_selected_tetrahedra = Vec::<[u32; 4]>::new();
                    for candidate_face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
                        let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
                        let conflicts = if boundary_faces.contains(&candidate_face) {
                            count != 0
                        } else {
                            count >= 2
                        };
                        if conflicts {
                            conflicting_faces.push(candidate_face);
                            if let Some(selected_tetrahedra) =
                                selected_tetrahedra_by_face.get(&candidate_face)
                            {
                                blocking_selected_tetrahedra
                                    .extend(selected_tetrahedra.iter().copied());
                            }
                        }
                    }
                    if !conflicting_faces.is_empty() {
                        blocking_selected_tetrahedra.sort();
                        blocking_selected_tetrahedra.dedup();
                        blockers.push(BoundaryExactCoverFaceCountBlocker {
                            node_ids: tetrahedron.node_ids,
                            exact_scaled_jacobian: tetrahedron.exact_scaled_jacobian,
                            conflicting_faces,
                            blocking_selected_tetrahedra,
                        });
                    }
                }
            }
        }
    }
    blockers.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverFaceCountBlockers {
        target_face,
        selected_tetrahedron_count: selected_tetrahedron_node_ids.len(),
        candidate_count,
        blocker_count: blockers.len(),
        blockers,
    })
}

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
    let mut selected_tetrahedra_by_face = BTreeMap::<[u32; 3], Vec<[u32; 4]>>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            selected_tetrahedra_by_face
                .entry(face)
                .or_default()
                .push(*selected_tetrahedron);
        }
    }
    let saturated_faces = selected_tetrahedra_by_face
        .iter()
        .filter_map(|(face, selected_tetrahedra)| {
            (!boundary_faces.contains(face) && selected_tetrahedra.len() >= 2).then_some(*face)
        })
        .collect::<BTreeSet<_>>();
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
    BoundaryExactCoverSaturatedComponent {
        seed_face,
        saturated_face_count: saturated_faces.len(),
        component_face_count: component_faces.len(),
        component_tetrahedron_count: component_tetrahedra.len(),
        component_faces: component_faces.into_iter().collect(),
        component_tetrahedra: component_tetrahedra.into_iter().collect(),
    }
}
