use super::*;

#[cfg(test)]
pub(super) fn exact_cover_trace_faces_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 3]>> {
    trace
        .dead_end_faces_by_reason
        .iter()
        .map(|(reason, faces)| (*reason, faces.iter().copied().collect::<Vec<_>>()))
        .collect()
}

#[cfg(test)]
pub(super) fn exact_cover_trace_selected_tetrahedra_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<[u32; 4]>> {
    let mut selected_tetrahedra_by_reason = BTreeMap::<&'static str, Vec<[u32; 4]>>::new();
    for dead_end in &trace.dead_ends {
        selected_tetrahedra_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_tetrahedra.clone());
    }
    selected_tetrahedra_by_reason
}

#[cfg(test)]
pub(super) fn exact_cover_trace_selected_roles_by_reason(
    trace: &BoundaryExactCoverTrace,
) -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut selected_roles_by_reason = BTreeMap::<&'static str, Vec<&'static str>>::new();
    for dead_end in &trace.dead_ends {
        selected_roles_by_reason
            .entry(dead_end.reason)
            .or_insert_with(|| dead_end.selected_roles.clone());
    }
    selected_roles_by_reason
}

#[cfg(test)]
pub(super) fn diagnostic_unforced_exact_cover_for_candidates(
    cavity: &ConstrainedCavity,
    candidates: &[ConstrainedCavityRefillTetrahedron],
    volume_relative_tolerance: f64,
) -> (bool, usize, usize, BTreeMap<&'static str, usize>) {
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        candidates,
        volume_relative_tolerance,
        250,
    );
    let (selected, trace) = search.search_without_forced_with_trace();
    (
        selected.is_some(),
        selected.map(|selected| selected.len()).unwrap_or(0),
        search.attempts,
        trace.dead_end_reason_counts,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundaryExactCoverDiagnostic {
        boundary_node_count: node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        solid_candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        zero_candidate_boundary_faces: Vec::new(),
        min_boundary_face_candidate_count: 0,
        min_candidate_boundary_faces: Vec::new(),
        max_boundary_face_candidate_count: 0,
        zero_solid_candidate_boundary_face_count: 0,
        zero_solid_candidate_boundary_faces: Vec::new(),
        min_solid_boundary_face_candidate_count: 0,
        min_solid_candidate_boundary_faces: Vec::new(),
        max_solid_boundary_face_candidate_count: 0,
        zero_addable_boundary_face_count: 0,
        zero_addable_boundary_faces: Vec::new(),
        min_addable_boundary_face_candidate_count: 0,
        min_addable_candidate_boundary_faces: Vec::new(),
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
    };
    if node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut solid_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
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
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
                    ) {
                        candidates.push(tetrahedron);
                    }
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        options,
                    ) {
                        solid_candidates.push(tetrahedron);
                    }
                }
            }
        }
    }
    diagnostic.candidate_count = candidates.len();
    diagnostic.solid_candidate_count = solid_candidates.len();
    let face_candidate_counts = boundary_faces
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
    let solid_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            solid_candidates
                .iter()
                .filter(|candidate| {
                    tetrahedron_faces(candidate.node_ids)
                        .map(sorted_face)
                        .contains(face)
                })
                .count()
        })
        .collect::<Vec<_>>();
    diagnostic.zero_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.min_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    diagnostic.zero_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_solid_candidate_boundary_face_count = solid_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_solid_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(solid_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_solid_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    diagnostic.max_solid_boundary_face_candidate_count = solid_face_candidate_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let root_face_counts = BTreeMap::<[u32; 3], usize>::new();
    let root_selected = Vec::<usize>::new();
    let addable_face_candidate_counts = boundary_faces
        .iter()
        .map(|face| {
            search.addable_candidate_count_for_face(*face, &root_face_counts, &root_selected)
        })
        .collect::<Vec<_>>();
    diagnostic.zero_addable_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| (*count == 0).then_some(*face))
        .collect();
    diagnostic.zero_addable_boundary_face_count = addable_face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_addable_boundary_face_candidate_count = addable_face_candidate_counts
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    diagnostic.min_addable_candidate_boundary_faces = boundary_faces
        .iter()
        .zip(addable_face_candidate_counts.iter())
        .filter_map(|(face, count)| {
            (*count == diagnostic.min_addable_boundary_face_candidate_count).then_some(*face)
        })
        .collect();
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.dead_end_reason = dead_end.reason;
        diagnostic.dead_end_face = dead_end.face;
        diagnostic.dead_end_depth = dead_end.depth;
        diagnostic.dead_end_selected_tetrahedra = dead_end.selected_tetrahedra;
        diagnostic.dead_end_current_volume_m3 = dead_end.current_volume_m3;
        diagnostic.dead_end_candidate_volume_m3 = dead_end.candidate_volume_m3;
        diagnostic.dead_end_target_volume_m3 = dead_end.target_volume_m3;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_mates_for_face(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    selected_tetrahedron_node_ids: &[[u32; 4]],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverMateDiagnostic, ConstrainedCavityRefillError> {
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
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
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
                    if let Ok(tetrahedron) = raw_refill_tetrahedron_with_rejection_reason(
                        tetrahedron_node_ids,
                        points,
                        relaxed_options,
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
pub(crate) fn diagnostic_boundary_exact_cover_face_candidate_sources(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    target_face: [u32; 3],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverFaceCandidateSourceDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let target_face = sorted_face(target_face);
    let relaxed_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..options
    };
    let mut diagnostic = BoundaryExactCoverFaceCandidateSourceDiagnostic {
        target_face,
        fourth_node_count: 0,
        centroid_inside_count: 0,
        solid_pass_count: 0,
        relaxed_pass_count: 0,
        outside_surface_count: 0,
        solid_rejected_by_reason: BTreeMap::new(),
        relaxed_rejected_by_reason: BTreeMap::new(),
        relaxed_candidate_node_ids: Vec::new(),
    };
    let face_nodes = target_face
        .map(|node_id| boundary_node_map.get(&node_id).copied())
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(ConstrainedCavityRefillError::MissingBoundaryNode {
            node_id: target_face[0],
        })?;
    for fourth_node_id in cavity_boundary_node_ids(cavity) {
        if target_face.contains(&fourth_node_id) {
            continue;
        }
        let Some(fourth_point) = boundary_node_map.get(&fourth_node_id).copied() else {
            return Err(ConstrainedCavityRefillError::MissingBoundaryNode {
                node_id: fourth_node_id,
            });
        };
        diagnostic.fourth_node_count += 1;
        let node_ids = [
            target_face[0],
            target_face[1],
            target_face[2],
            fourth_node_id,
        ];
        let points = [face_nodes[0], face_nodes[1], face_nodes[2], fourth_point];
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            diagnostic.outside_surface_count += 1;
            continue;
        }
        diagnostic.centroid_inside_count += 1;
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, options) {
            Ok(_) => diagnostic.solid_pass_count += 1,
            Err(reason) => {
                *diagnostic
                    .solid_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
        match raw_refill_tetrahedron_with_rejection_reason(node_ids, points, relaxed_options) {
            Ok(tetrahedron) => {
                diagnostic.relaxed_pass_count += 1;
                diagnostic
                    .relaxed_candidate_node_ids
                    .push(sorted_tetrahedron_nodes(tetrahedron.node_ids));
            }
            Err(reason) => {
                *diagnostic
                    .relaxed_rejected_by_reason
                    .entry(reason)
                    .or_default() += 1
            }
        }
    }
    diagnostic.relaxed_candidate_node_ids.sort();
    diagnostic.relaxed_candidate_node_ids.dedup();
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_interior_mate_closure(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
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
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
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
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    if touches_boundary
                        && candidate_keys.insert(sorted_tetrahedron_nodes(tetrahedron.node_ids))
                    {
                        candidates.push(tetrahedron.clone());
                    }
                    all_candidates.push(tetrahedron);
                }
            }
        }
    }
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    for _ in 0..4 {
        let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
        for candidate in &candidates {
            for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
                *face_counts.entry(face).or_default() += 1;
            }
        }
        let missing_faces = face_counts
            .iter()
            .filter_map(|(face, count)| {
                (!boundary_faces.contains(face) && *count == 1).then_some(*face)
            })
            .collect::<Vec<_>>();
        if missing_faces.is_empty() {
            break;
        }
        let mut added = false;
        for face in missing_faces {
            if let Some(indices) = all_candidates_by_face.get(&face) {
                let mut indices = indices.clone();
                indices.sort_by(|left, right| {
                    all_candidates[*right]
                        .exact_scaled_jacobian
                        .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
                });
                for index in indices {
                    if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                        break;
                    }
                    let candidate = &all_candidates[index];
                    if candidate_keys.insert(sorted_tetrahedron_nodes(candidate.node_ids)) {
                        candidates.push(candidate.clone());
                        added = true;
                        break;
                    }
                }
            }
        }
        if !added {
            break;
        }
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let (selected, trace) = search.search_with_trace();
    let Some(selected) = selected else {
        let dead_end = trace.dead_end.clone();
        return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
            initial_candidate_count,
            candidate_count: candidates.len(),
            injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
            found_cover: false,
            selected_tetrahedron_count: 0,
            search_attempt_count: search.attempts,
            reason: if search.attempts > 5_000 {
                "search_exhausted"
            } else {
                "cover_not_found"
            },
            dead_end_reason: dead_end
                .as_ref()
                .map(|dead_end| dead_end.reason)
                .unwrap_or("not_evaluated"),
            dead_end_face: dead_end.as_ref().and_then(|dead_end| dead_end.face),
            dead_end_depth: dead_end
                .as_ref()
                .map(|dead_end| dead_end.depth)
                .unwrap_or(0),
            dead_end_selected_tetrahedra: dead_end
                .as_ref()
                .map(|dead_end| dead_end.selected_tetrahedra.clone())
                .unwrap_or_default(),
            dead_end_current_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.current_volume_m3)
                .unwrap_or(0.0),
            dead_end_candidate_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.candidate_volume_m3)
                .unwrap_or(0.0),
            dead_end_target_volume_m3: dead_end
                .as_ref()
                .map(|dead_end| dead_end.target_volume_m3)
                .unwrap_or(0.0),
            dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
            dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
            dead_end_selected_tetrahedra_by_reason: exact_cover_trace_selected_tetrahedra_by_reason(
                &trace,
            ),
            dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(&trace),
            unforced_found_cover: false,
            unforced_selected_tetrahedron_count: 0,
            unforced_search_attempt_count: 0,
            unforced_dead_end_reason_histogram: BTreeMap::new(),
        });
    };
    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: true,
        selected_tetrahedron_count: selected.len(),
        search_attempt_count: search.attempts,
        reason: "cover_found",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
        cavity,
        boundary_nodes,
        &[],
        options,
    )
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    excluded_tetrahedron_node_ids: &[[u32; 4]],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
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
    let mut all_candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    let mut candidate_keys = BTreeSet::<[u32; 4]>::new();
    let excluded_keys = excluded_tetrahedron_node_ids
        .iter()
        .copied()
        .map(sorted_tetrahedron_nodes)
        .collect::<BTreeSet<_>>();
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
                    let touches_boundary = tetrahedron_faces(tetrahedron.node_ids)
                        .map(sorted_face)
                        .iter()
                        .any(|face| boundary_faces.contains(face));
                    let candidate_key = sorted_tetrahedron_nodes(tetrahedron.node_ids);
                    if touches_boundary
                        && !excluded_keys.contains(&candidate_key)
                        && candidate_keys.insert(candidate_key)
                    {
                        candidates.push(tetrahedron.clone());
                    }
                    all_candidates.push(tetrahedron);
                }
            }
        }
    }
    let initial_candidate_count = candidates.len();
    let mut all_candidates_by_face = BTreeMap::<[u32; 3], Vec<usize>>::new();
    for (index, candidate) in all_candidates.iter().enumerate() {
        for face in tetrahedron_faces(candidate.node_ids).map(sorted_face) {
            all_candidates_by_face.entry(face).or_default().push(index);
        }
    }
    let mut total_attempts = 0_usize;
    for _ in 0..64 {
        let mut search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let (selected, trace) = search.search_with_trace();
        total_attempts += search.attempts;
        if let Some(selected) = selected {
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: true,
                selected_tetrahedron_count: selected.len(),
                search_attempt_count: total_attempts,
                reason: "cover_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover: false,
                unforced_selected_tetrahedron_count: 0,
                unforced_search_attempt_count: 0,
                unforced_dead_end_reason_histogram: BTreeMap::new(),
            });
        }
        let Some(dead_end) = trace.dead_end.clone() else {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: "cover_not_found",
                dead_end_reason: "not_evaluated",
                dead_end_face: None,
                dead_end_depth: 0,
                dead_end_selected_tetrahedra: Vec::new(),
                dead_end_current_volume_m3: 0.0,
                dead_end_candidate_volume_m3: 0.0,
                dead_end_target_volume_m3: 0.0,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        };
        let future_mate_dead_ends = trace
            .dead_ends
            .iter()
            .filter(|dead_end| dead_end.reason == "forced_interior_mate_future_mate_conflict")
            .cloned()
            .collect::<Vec<_>>();
        let no_candidate_dead_end_faces = trace
            .dead_ends
            .iter()
            .filter_map(|dead_end| {
                (dead_end.reason == "forced_interior_mate_no_candidate_contains_face")
                    .then_some(dead_end.face)
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        if future_mate_dead_ends.is_empty() && no_candidate_dead_end_faces.is_empty() {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
        let search =
            BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
        let mut mate_faces = BTreeSet::<[u32; 3]>::new();
        mate_faces.extend(no_candidate_dead_end_faces);
        for future_dead_end in &future_mate_dead_ends {
            let Some(face) = future_dead_end.face else {
                continue;
            };
            let selected_indices = future_dead_end
                .selected_tetrahedra
                .iter()
                .filter_map(|selected_tetrahedron| {
                    candidates.iter().position(|candidate| {
                        sorted_tetrahedron_nodes(candidate.node_ids)
                            == sorted_tetrahedron_nodes(*selected_tetrahedron)
                    })
                })
                .collect::<Vec<_>>();
            let mut face_counts = BTreeMap::<[u32; 3], usize>::new();
            for selected_index in &selected_indices {
                for selected_face in
                    tetrahedron_faces(candidates[*selected_index].node_ids).map(sorted_face)
                {
                    *face_counts.entry(selected_face).or_default() += 1;
                }
            }
            for candidate_index in (0..candidates.len()).filter(|candidate_index| {
                !selected_indices.contains(candidate_index)
                    && search.candidate_faces[*candidate_index].contains(&face)
            }) {
                for candidate_face in search.candidate_faces[candidate_index] {
                    if !boundary_faces.contains(&candidate_face)
                        && face_counts.get(&candidate_face).copied().unwrap_or(0) == 0
                        && !search.interior_face_has_future_mate(
                            candidate_index,
                            candidate_face,
                            &face_counts,
                            &selected_indices,
                        )
                    {
                        mate_faces.insert(candidate_face);
                    }
                }
            }
        }
        let mut added = false;
        for mate_face in mate_faces {
            let Some(indices) = all_candidates_by_face.get(&mate_face) else {
                continue;
            };
            let mut indices = indices.clone();
            indices.sort_by(|left, right| {
                all_candidates[*right]
                    .exact_scaled_jacobian
                    .total_cmp(&all_candidates[*left].exact_scaled_jacobian)
            });
            for index in indices {
                if candidates.len() >= MAX_BOUNDARY_EXACT_COVER_CANDIDATES {
                    break;
                }
                let candidate = &all_candidates[index];
                let candidate_key = sorted_tetrahedron_nodes(candidate.node_ids);
                if !excluded_keys.contains(&candidate_key) && candidate_keys.insert(candidate_key) {
                    candidates.push(candidate.clone());
                    added = true;
                    break;
                }
            }
        }
        if !added {
            let (
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            ) = diagnostic_unforced_exact_cover_for_candidates(
                cavity,
                &candidates,
                options.volume_relative_tolerance,
            );
            return Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
                initial_candidate_count,
                candidate_count: candidates.len(),
                injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
                found_cover: false,
                selected_tetrahedron_count: 0,
                search_attempt_count: total_attempts,
                reason: if total_attempts > 5_000 {
                    "search_exhausted"
                } else {
                    "cover_not_found"
                },
                dead_end_reason: dead_end.reason,
                dead_end_face: dead_end.face,
                dead_end_depth: dead_end.depth,
                dead_end_selected_tetrahedra: dead_end.selected_tetrahedra,
                dead_end_current_volume_m3: dead_end.current_volume_m3,
                dead_end_candidate_volume_m3: dead_end.candidate_volume_m3,
                dead_end_target_volume_m3: dead_end.target_volume_m3,
                dead_end_reason_histogram: trace.dead_end_reason_counts.clone(),
                dead_end_faces_by_reason: exact_cover_trace_faces_by_reason(&trace),
                dead_end_selected_tetrahedra_by_reason:
                    exact_cover_trace_selected_tetrahedra_by_reason(&trace),
                dead_end_selected_roles_by_reason: exact_cover_trace_selected_roles_by_reason(
                    &trace,
                ),
                unforced_found_cover,
                unforced_selected_tetrahedron_count,
                unforced_search_attempt_count,
                unforced_dead_end_reason_histogram,
            });
        }
    }

    Ok(BoundaryExactCoverInteriorMateClosureDiagnostic {
        initial_candidate_count,
        candidate_count: candidates.len(),
        injected_candidate_count: candidates.len().saturating_sub(initial_candidate_count),
        found_cover: false,
        selected_tetrahedron_count: 0,
        search_attempt_count: total_attempts,
        reason: "iteration_limit",
        dead_end_reason: "not_evaluated",
        dead_end_face: None,
        dead_end_depth: 0,
        dead_end_selected_tetrahedra: Vec::new(),
        dead_end_current_volume_m3: 0.0,
        dead_end_candidate_volume_m3: 0.0,
        dead_end_target_volume_m3: 0.0,
        dead_end_reason_histogram: BTreeMap::new(),
        dead_end_faces_by_reason: BTreeMap::new(),
        dead_end_selected_tetrahedra_by_reason: BTreeMap::new(),
        dead_end_selected_roles_by_reason: BTreeMap::new(),
        unforced_found_cover: false,
        unforced_selected_tetrahedron_count: 0,
        unforced_search_attempt_count: 0,
        unforced_dead_end_reason_histogram: BTreeMap::new(),
    })
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundarySteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let mut diagnostic = BoundarySteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }
    let Some(centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };
    if point_in_closed_triangle_surface(centroid, &boundary_triangles, MeshingTolerance::default())
        != PointInClosedSurface::Inside
    {
        diagnostic.reason = "steiner_point_outside_cavity";
        return Ok(diagnostic);
    }
    let steiner_node_id = next_cavity_node_id(cavity);
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    node_points.insert(steiner_node_id, centroid);
    let mut node_ids = boundary_node_ids.clone();
    node_ids.push(steiner_node_id);
    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                        .iter()
                        .any(|face| boundary_faces.contains(face))
                    {
                        continue;
                    }
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
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
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
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
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 512 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
}

#[cfg(test)]
pub(crate) fn diagnostic_boundary_patch_steiner_exact_cover(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryPatchSteinerExactCoverDiagnostic, ConstrainedCavityRefillError> {
    validate_refill_options(options)?;
    validate_constrained_cavity(cavity).map_err(ConstrainedCavityRefillError::Validation)?;
    let boundary_node_map = boundary_node_coordinates(cavity, boundary_nodes)?;
    let boundary_triangles = cavity_boundary_triangles(cavity, &boundary_node_map)?;
    let boundary_node_ids = cavity_boundary_node_ids(cavity)
        .into_iter()
        .collect::<Vec<_>>();
    let boundary_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| sorted_face(face.node_ids))
        .collect::<BTreeSet<_>>();
    let mut diagnostic = BoundaryPatchSteinerExactCoverDiagnostic {
        boundary_node_count: boundary_node_ids.len(),
        boundary_face_count: cavity.boundary_faces.len(),
        missing_face_count: 0,
        patch_count: 0,
        steiner_node_count: 0,
        candidate_count: 0,
        zero_candidate_boundary_face_count: 0,
        min_boundary_face_candidate_count: 0,
        max_boundary_face_candidate_count: 0,
        selected_tetrahedron_count: 0,
        search_attempt_count: 0,
        found_cover: false,
        reason: "not_evaluated",
        max_min_scaled_jacobian: 0.0,
    };
    if boundary_node_ids.len() < 4 {
        diagnostic.reason = "too_few_boundary_nodes";
        return Ok(diagnostic);
    }

    let points = boundary_node_ids
        .iter()
        .map(|node_id| ConnectivityPoint {
            node_id: *node_id,
            coordinates_m: boundary_node_map[node_id],
            is_super: false,
        })
        .collect::<Vec<_>>();
    let mut boundary_refill_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for tetrahedron in tetrahedralize_points(&points) {
        let node_ids = tetrahedron.vertices.map(|index| points[index].node_id);
        let tetrahedron_points = tetrahedron
            .vertices
            .map(|index| points[index].coordinates_m);
        if point_in_closed_triangle_surface(
            tetrahedron_centroid(tetrahedron_points),
            &boundary_triangles,
            MeshingTolerance::default(),
        ) != PointInClosedSurface::Inside
        {
            continue;
        }
        if let Ok(tetrahedron) =
            raw_refill_tetrahedron_with_rejection_reason(node_ids, tetrahedron_points, options)
        {
            boundary_refill_tetrahedra.push(tetrahedron);
        }
    }
    let missing_faces = missing_refill_boundary_faces(cavity, &boundary_refill_tetrahedra)
        .map_err(ConstrainedCavityRefillError::Validation)?;
    diagnostic.missing_face_count = missing_faces.len();
    if missing_faces.is_empty() {
        diagnostic.reason = "no_missing_faces";
        return Ok(diagnostic);
    }
    let Some(cavity_centroid) = cavity_boundary_node_centroid(cavity, &boundary_node_map) else {
        diagnostic.reason = "empty_boundary";
        return Ok(diagnostic);
    };

    let components = missing_face_components(&missing_faces, MissingFaceLink::Node);
    diagnostic.patch_count = components.len();
    let mut node_points = boundary_node_ids
        .iter()
        .map(|node_id| (*node_id, boundary_node_map[node_id]))
        .collect::<BTreeMap<_, _>>();
    let mut node_ids = boundary_node_ids.clone();
    let mut next_node_id = next_cavity_node_id(cavity);
    for component in components {
        let mut patch_node_ids = BTreeSet::<u32>::new();
        for face_index in component {
            patch_node_ids.extend(missing_faces[face_index]);
        }
        let Some(surface_point) = centroid_of_node_set(&patch_node_ids, &boundary_node_map) else {
            continue;
        };
        let Some(point) =
            patch_steiner_point_inside_cavity(surface_point, cavity_centroid, &boundary_triangles)
        else {
            continue;
        };
        while node_points.contains_key(&next_node_id) {
            next_node_id = next_node_id.saturating_add(1);
        }
        node_points.insert(next_node_id, point);
        node_ids.push(next_node_id);
        diagnostic.steiner_node_count += 1;
        next_node_id = next_node_id.saturating_add(1);
    }
    if diagnostic.steiner_node_count == 0 {
        diagnostic.reason = "no_valid_patch_steiner_points";
        return Ok(diagnostic);
    }

    let mut candidates = Vec::<ConstrainedCavityRefillTetrahedron>::new();
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
                    let points = tetrahedron_node_ids.map(|node_id| node_points[&node_id]);
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
    diagnostic.candidate_count = candidates.len();
    let face_candidate_counts = boundary_faces
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
    diagnostic.zero_candidate_boundary_face_count = face_candidate_counts
        .iter()
        .filter(|count| **count == 0)
        .count();
    diagnostic.min_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().min().unwrap_or(0);
    diagnostic.max_boundary_face_candidate_count =
        face_candidate_counts.iter().copied().max().unwrap_or(0);
    if candidates.is_empty() {
        diagnostic.reason = "no_candidate_tetrahedra";
        return Ok(diagnostic);
    }
    if candidates.len() > 1_024 {
        diagnostic.reason = "over_candidate_limit";
        return Ok(diagnostic);
    }
    let mut search =
        BoundaryExactCoverSearch::new(cavity, &candidates, options.volume_relative_tolerance);
    let selected = search.search();
    diagnostic.search_attempt_count = search.attempts;
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > 5_000 {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return Ok(diagnostic);
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidates[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    Ok(diagnostic)
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
