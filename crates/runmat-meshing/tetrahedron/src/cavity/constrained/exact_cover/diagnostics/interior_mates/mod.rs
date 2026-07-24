use super::*;

#[cfg(test)]
mod closure;

#[cfg(test)]
mod on_demand;

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
pub(crate) fn diagnostic_boundary_exact_cover_with_on_demand_interior_mates(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    options: ConstrainedCavityRefillOptions,
) -> Result<BoundaryExactCoverInteriorMateClosureDiagnostic, ConstrainedCavityRefillError> {
    on_demand::diagnostic_boundary_exact_cover_with_on_demand_interior_mates(
        cavity,
        boundary_nodes,
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
    on_demand::diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
        cavity,
        boundary_nodes,
        excluded_tetrahedron_node_ids,
        options,
    )
}
