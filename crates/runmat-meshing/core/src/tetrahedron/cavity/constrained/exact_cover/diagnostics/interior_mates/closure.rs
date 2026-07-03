use super::*;

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
