use super::*;

pub(super) fn emit_progress_checkpoint(
    insertion_count: u64,
    topology: &crate::cdt::DelaunayVolumeTopology,
    quality: &crate::cdt::DelaunayVolumeQuality,
    state: &'static str,
) {
    let violation_count = quality
        .tetrahedra
        .iter()
        .filter(|tetrahedron| tetrahedron.requires_refinement())
        .count();
    let worst = quality.tetrahedra.iter().max_by(|left, right| {
        left.refinement_violation_ratio
            .total_cmp(&right.refinement_violation_ratio)
            .then_with(|| right.node_identities.cmp(&left.node_identities))
    });
    let worst_violation_ratio = worst.map_or(0.0, |cell| cell.refinement_violation_ratio);
    let worst_violation_count = quality
        .tetrahedra
        .iter()
        .filter(|cell| cell.refinement_violation_ratio.to_bits() == worst_violation_ratio.to_bits())
        .count();
    let boundary_nodes = topology
        .incidence
        .boundary_facets
        .iter()
        .flat_map(|facet| facet.vertex_indices)
        .collect::<BTreeSet<_>>();
    let worst_boundary_node_count = worst.map_or(0, |cell| {
        cell.node_identities
            .iter()
            .filter(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(identity, |node| &node.identity)
                    .is_ok_and(|index| boundary_nodes.contains(&(index as u32)))
            })
            .count()
    });
    let worst_maximum_metric_edge_length =
        worst.map_or(0.0, |cell| cell.maximum_metric_edge_length);
    let worst_metric_radius_edge_ratio = worst.map_or(0.0, |cell| cell.metric_radius_edge_ratio);
    let worst_metric_scaled_jacobian = worst.map_or(0.0, |cell| cell.metric_scaled_jacobian);
    let span = tracing::info_span!(
        "meshing.tetrahedron.refinement_checkpoint",
        state,
        insertion_count,
        node_count = topology.nodes.len(),
        tetrahedron_count = topology.tetrahedra.len(),
        violation_count,
        worst_violation_ratio,
        worst_violation_count,
        worst_boundary_node_count,
        worst_maximum_metric_edge_length,
        worst_metric_radius_edge_ratio,
        worst_metric_scaled_jacobian,
    );
    let _guard = span.enter();
    tracing::info!(
        state,
        insertion_count,
        node_count = topology.nodes.len(),
        tetrahedron_count = topology.tetrahedra.len(),
        violation_count,
        worst_violation_ratio,
        worst_violation_count,
        worst_boundary_node_count,
        worst_maximum_metric_edge_length,
        worst_metric_radius_edge_ratio,
        worst_metric_scaled_jacobian,
        "tetrahedral refinement checkpoint"
    );
}
