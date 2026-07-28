use super::*;

mod candidates;
mod excluding;

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
    excluding::diagnostic_boundary_exact_cover_with_on_demand_interior_mates_excluding(
        cavity,
        boundary_nodes,
        excluded_tetrahedron_node_ids,
        options,
    )
}
