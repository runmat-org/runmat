use super::*;

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    if !tetrahedron_faces(tetrahedron_node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    let search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected_keys = selected_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
    let selected = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            selected_keys
                .contains(&sorted_tetrahedron_nodes(candidate.node_ids))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
    for candidate_index in &selected {
        for face in search.candidate_faces[*candidate_index] {
            *face_counts.entry(face).or_default() += 1;
        }
    }

    let target_face = sorted_face(target_face);
    let mut diagnostics = Vec::<BoundaryExactCoverMateCandidateDiagnostic>::new();
    for candidate_index in 0..candidates.len() {
        if selected.contains(&candidate_index)
            || !search.candidate_faces[candidate_index].contains(&target_face)
        {
            continue;
        }
        let mut conflicting_faces = Vec::<[u32; 3]>::new();
        let mut missing_future_mate_faces = Vec::<[u32; 3]>::new();
        for candidate_face in search.candidate_faces[candidate_index] {
            let count = face_counts.get(&candidate_face).copied().unwrap_or(0);
            if if boundary_faces.contains(&candidate_face) {
                count != 0
            } else {
                count >= 2
            } {
                conflicting_faces.push(candidate_face);
            }
            if !boundary_faces.contains(&candidate_face)
                && count == 0
                && !search.interior_face_has_future_mate(
                    candidate_index,
                    candidate_face,
                    &face_counts,
                    &selected,
                )
            {
                missing_future_mate_faces.push(candidate_face);
            }
        }
        let addable = search.candidate_can_be_added_for_face(
            candidate_index,
            target_face,
            &face_counts,
            &selected,
        );
        diagnostics.push(BoundaryExactCoverMateCandidateDiagnostic {
            node_ids: candidates[candidate_index].node_ids,
            exact_scaled_jacobian: candidates[candidate_index].exact_scaled_jacobian,
            addable,
            conflicting_faces,
            missing_future_mate_faces,
        });
    }
    diagnostics.sort_by(|left, right| left.node_ids.cmp(&right.node_ids));
    Ok(BoundaryExactCoverMateDiagnostic {
        target_face,
        candidate_count: diagnostics.len(),
        addable_count: diagnostics
            .iter()
            .filter(|candidate| candidate.addable)
            .count(),
        candidates: diagnostics,
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_support_node_exact_cover(
    cavity: &ConstrainedCavity,
    nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<SupportNodeExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, nodes)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity);
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let mut node_map = BTreeMap::<u32, Point3>::new();
    for node in nodes {
        if node_map.insert(node.node_id, node.coordinates_m).is_some() {
            return Err(ConstrainedCavityRefillError::DuplicateInteriorNode {
                node_id: node.node_id,
            });
        }
    }
    let mut candidate_nodes = Vec::<ConstrainedCavityNode>::new();
    for node in nodes {
        if boundary_node_ids.contains(&node.node_id) {
            candidate_nodes.push(node.clone());
            continue;
        }
        if !candidate_respects_protected_boundary_distance(
            cavity,
            &boundary_node_map,
            node.coordinates_m,
            options,
        ) {
            continue;
        }
        if point_in_closed_triangle_surface(
            node.coordinates_m,
            &boundary_triangles,
            MeshingTolerance::default(),
        ) == PointInClosedSurface::Inside
        {
            candidate_nodes.push(node.clone());
        }
    }
    candidate_nodes.sort_by_key(|node| node.node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut seen_tetrahedra = BTreeSet::<[u32; 4]>::new();
    for first in 0..candidate_nodes.len() {
        for second in (first + 1)..candidate_nodes.len() {
            for third in (second + 1)..candidate_nodes.len() {
                for fourth in (third + 1)..candidate_nodes.len() {
                    let tetrahedron_node_ids = [
                        candidate_nodes[first].node_id,
                        candidate_nodes[second].node_id,
                        candidate_nodes[third].node_id,
                        candidate_nodes[fourth].node_id,
                    ];
                    if !seen_tetrahedra.insert(sorted_tetrahedron_nodes(tetrahedron_node_ids)) {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_map[&node_id]);
                    if point_in_closed_triangle_surface(
                        tetrahedron_centroid(points),
                        &boundary_triangles,
                        MeshingTolerance::default(),
                    ) != PointInClosedSurface::Inside
                    {
                        continue;
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    if candidates.is_empty() {
        return Ok(SupportNodeExactCoverDiagnostic {
            candidate_node_count: candidate_nodes.len(),
            candidate_count: 0,
            root_zero_raw_boundary_face_count: 0,
            root_zero_raw_boundary_faces: Vec::new(),
            root_min_raw_boundary_face_candidate_count: 0,
            root_min_raw_candidate_boundary_faces: Vec::new(),
            root_max_raw_boundary_face_candidate_count: 0,
            root_zero_addable_boundary_face_count: 0,
            root_zero_addable_boundary_faces: Vec::new(),
            root_min_addable_boundary_face_candidate_count: 0,
            root_min_addable_candidate_boundary_faces: Vec::new(),
            root_max_addable_boundary_face_candidate_count: 0,
            selected_tetrahedron_count: 0,
            search_attempt_count: 0,
            found_cover: false,
            reason: "no_candidate_tetrahedra",
            dead_end_reason: "not_evaluated",
            dead_end_face: None,
            dead_end_depth: 0,
            dead_end_reason_histogram: BTreeMap::new(),
            dead_end_faces_by_reason: BTreeMap::new(),
        });
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let root_raw_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    let root_zero_raw_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_raw_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_raw_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_raw_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_raw_boundary_face_candidate_count = root_raw_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let root_addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    let root_zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect::<Vec<_>>();
    let root_min_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let root_min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(root_addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == root_min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect::<Vec<_>>();
    let root_max_addable_boundary_face_candidate_count = root_addable_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let (selected, trace) = search.search_with_trace();
    let dead_end = trace.dead_end.clone();
    let dead_end_faces_by_reason = exact_cover_trace_faces_by_reason(&trace);
    Ok(SupportNodeExactCoverDiagnostic {
        candidate_node_count: candidate_nodes.len(),
        candidate_count: candidates.len(),
        root_zero_raw_boundary_face_count: root_zero_raw_boundary_faces.len(),
        root_zero_raw_boundary_faces,
        root_min_raw_boundary_face_candidate_count,
        root_min_raw_candidate_boundary_faces,
        root_max_raw_boundary_face_candidate_count,
        root_zero_addable_boundary_face_count: root_zero_addable_boundary_faces.len(),
        root_zero_addable_boundary_faces,
        root_min_addable_boundary_face_candidate_count,
        root_min_addable_candidate_boundary_faces,
        root_max_addable_boundary_face_candidate_count,
        selected_tetrahedron_count: selected.as_ref().map(Vec::len).unwrap_or(0),
        search_attempt_count: search.attempts,
        found_cover: selected.is_some(),
        reason: if selected.is_some() {
            "cover_found"
        } else if search.attempts > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        },
        dead_end_reason: dead_end
            .as_ref()
            .map(|dead_end| dead_end.reason)
            .unwrap_or("not_evaluated"),
        dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
        dead_end_depth: dead_end.map(|dead_end| dead_end.depth).unwrap_or(0),
        dead_end_reason_histogram: trace.dead_end_reason_counts,
        dead_end_faces_by_reason,
    })
}
