use super::*;

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
