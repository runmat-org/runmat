use super::*;

mod candidates;

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
    let candidates::BoundaryExactCoverCandidateSummary {
        boundary_faces,
        candidates,
        solid_candidates,
        face_candidate_counts,
        solid_face_candidate_counts,
    } = candidates::boundary_exact_cover_candidate_summary(
        cavity,
        &node_ids,
        &boundary_node_map,
        &boundary_triangles,
        options,
    );
    diagnostic.candidate_count = candidates.len();
    diagnostic.solid_candidate_count = solid_candidates.len();
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
