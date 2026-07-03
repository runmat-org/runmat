use super::*;

pub(crate) mod interior_mates;
mod steiner;
pub(crate) mod support_node;
pub(crate) use steiner::*;

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
