use super::*;

mod boundary;
pub(crate) mod face_candidates;
pub(crate) mod interior_mates;
mod steiner;
pub(crate) mod support_node;
pub(crate) use boundary::diagnostic_boundary_exact_cover;
pub(crate) use face_candidates::diagnostic_boundary_exact_cover_face_candidate_sources;
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
