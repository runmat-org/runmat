use super::*;

const SHARED_CAP_CANDIDATE_LIMIT: usize = 4_096;
const SHARED_CAP_SEARCH_ATTEMPT_LIMIT: usize = 25_000;

pub(super) fn finish_shared_cap_exact_cover_diagnostic(
    cavity: &ConstrainedCavity,
    candidate_tetrahedra: &[ConstrainedCavityRefillTetrahedron],
    options: ConstrainedCavityRefillOptions,
    mut diagnostic: MissingFaceLocalCapStitchDiagnostic,
) -> MissingFaceLocalCapStitchDiagnostic {
    if candidate_tetrahedra.len() > SHARED_CAP_CANDIDATE_LIMIT {
        diagnostic.reason = "over_candidate_limit";
        return diagnostic;
    }
    let mut search = BoundaryExactCoverSearch::with_attempt_limit(
        cavity,
        candidate_tetrahedra,
        options.volume_relative_tolerance,
        SHARED_CAP_SEARCH_ATTEMPT_LIMIT,
    );
    let root_availability = search.root_boundary_availability();
    diagnostic.root_boundary_zero_raw_candidate_face_count =
        root_availability.zero_raw_candidate_face_count;
    diagnostic.root_boundary_zero_addable_candidate_face_count =
        root_availability.zero_addable_candidate_face_count;
    diagnostic.root_boundary_min_raw_candidate_count = root_availability.min_raw_candidate_count;
    diagnostic.root_boundary_min_addable_candidate_count =
        root_availability.min_addable_candidate_count;
    diagnostic.root_boundary_max_addable_candidate_count =
        root_availability.max_addable_candidate_count;
    let (selected, trace) = search.search_with_trace();
    diagnostic.search_attempt_count = search.attempts;
    diagnostic.cover_dead_end_reason_histogram = trace.dead_end_reason_counts;
    if let Some(dead_end) = trace.dead_end {
        diagnostic.cover_dead_end_reason = dead_end.reason;
        diagnostic.cover_dead_end_depth = dead_end.depth;
    }
    let Some(selected) = selected else {
        diagnostic.reason = if diagnostic.search_attempt_count > SHARED_CAP_SEARCH_ATTEMPT_LIMIT {
            "search_exhausted"
        } else {
            "cover_not_found"
        };
        return diagnostic;
    };
    diagnostic.max_min_scaled_jacobian = selected
        .iter()
        .map(|index| candidate_tetrahedra[*index].exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    diagnostic.selected_tetrahedron_count = selected.len();
    diagnostic.found_cover = true;
    diagnostic.reason = "cover_found";
    diagnostic
}
