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
    let target_face = sorted_face(target_face);
    let selected_counts = selected_face_counts(selected_tetrahedron_node_ids);
    let selected_tetrahedra_by_face = selected_tetrahedra_by_face(selected_tetrahedron_node_ids);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };

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
                    if !candidate_matches_target_face(
                        tetrahedron_node_ids,
                        target_face,
                        &selected_keys,
                    ) {
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
                    if let Some(blocker) = face_count_blocker_for_candidate(
                        &tetrahedron,
                        &selected_counts,
                        &selected_tetrahedra_by_face,
                        &boundary_faces,
                    ) {
                        blockers.push(blocker);
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

fn selected_face_counts(selected_tetrahedron_node_ids: &[[u32; 4]]) -> BTreeMap<[u32; 3], usize> {
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for selected_tetrahedron in selected_tetrahedron_node_ids {
        for face in tetrahedron_faces(*selected_tetrahedron).map(sorted_face) {
            *face_counts.entry(face).or_default() += 1;
        }
    }
    face_counts
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

fn candidate_matches_target_face(
    tetrahedron_node_ids: [u32; 4],
    target_face: [u32; 3],
    selected_keys: &BTreeSet<[u32; 4]>,
) -> bool {
    tetrahedron_faces(tetrahedron_node_ids)
        .map(sorted_face)
        .contains(&target_face)
        && !selected_keys.contains(&sorted_tetrahedron_nodes(tetrahedron_node_ids))
}

fn face_count_blocker_for_candidate(
    tetrahedron: &ConstrainedCavityRefillTetrahedron,
    selected_counts: &BTreeMap<[u32; 3], usize>,
    selected_tetrahedra_by_face: &BTreeMap<[u32; 3], Vec<[u32; 4]>>,
    boundary_faces: &BTreeSet<[u32; 3]>,
) -> Option<BoundaryExactCoverFaceCountBlocker> {
    let mut conflicting_faces = Vec::<[u32; 3]>::new();
    let mut blocking_selected_tetrahedra = Vec::<[u32; 4]>::new();
    for candidate_face in tetrahedron_faces(tetrahedron.node_ids).map(sorted_face) {
        let count = selected_counts.get(&candidate_face).copied().unwrap_or(0);
        let conflicts = if boundary_faces.contains(&candidate_face) {
            count != 0
        } else {
            count >= 2
        };
        if conflicts {
            conflicting_faces.push(candidate_face);
            if let Some(selected_tetrahedra) = selected_tetrahedra_by_face.get(&candidate_face) {
                blocking_selected_tetrahedra.extend(selected_tetrahedra.iter().copied());
            }
        }
    }
    if conflicting_faces.is_empty() {
        return None;
    }
    blocking_selected_tetrahedra.sort();
    blocking_selected_tetrahedra.dedup();
    Some(BoundaryExactCoverFaceCountBlocker {
        node_ids: tetrahedron.node_ids,
        exact_scaled_jacobian: tetrahedron.exact_scaled_jacobian,
        conflicting_faces,
        blocking_selected_tetrahedra,
    })
}
