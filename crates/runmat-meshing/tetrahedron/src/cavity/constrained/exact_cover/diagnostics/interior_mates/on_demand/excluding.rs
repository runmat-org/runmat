use super::candidates::{build_on_demand_interior_mate_candidates, OnDemandInteriorMateCandidates};
use super::*;

mod mate_faces;

pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    excluded_tetrahedron_node_ids: &[[u32; 4]],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    let OnDemandInteriorMateCandidates {
        all_candidates,
        mut candidates,
        mut candidate_keys,
        excluded_keys,
        boundary_faces,
        all_candidates_by_face,
        initial_candidate_count,
    } = build_on_demand_interior_mate_candidates(
        cavity,
        boundary_nodes,
        excluded_tetrahedron_node_ids,
        options,
    )?;
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
        let mate_faces = mate_faces::on_demand_interior_mate_faces_for_trace(
            cavity,
            &candidates,
            options,
            &boundary_faces,
            no_candidate_dead_end_faces,
            &future_mate_dead_ends,
        );
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
