//! Bounded local evidence for failed polygon support. Never dump the full topology.
use super::*;
pub(super) fn report(
    recovery: &FacetSupportView<'_, impl TetrahedralTopology>,
    boundary: &[StableDigest],
    source: [StableDigest; 3],
    membership: &BTreeMap<StableDigest, BTreeSet<usize>>,
    projection: ([usize; 2], PredicateSign),
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<(), DelaunayFacetRecoveryError> {
    let (axes, sign) = projection;
    const MAXIMUM_WITNESS_NODES: usize = 64;
    let mut nodes = Vec::with_capacity(boundary.len().min(MAXIMUM_WITNESS_NODES));
    for &id in boundary.iter().take(MAXIMUM_WITNESS_NODES) {
        work.support_step(constraint_index)?;
        let node = node_index(recovery.topology, id)
            .ok_or_else(|| invalid(constraint_index, "facet witness node missing"))?;
        nodes.push((
            id,
            recovery.topology.nodes()[node]
                .coordinates_m
                .canonical_value()
                .map(f64::to_bits),
            recovery.topology.nodes()[node].construction,
            membership.get(&id),
        ));
    }
    let source_points = source.map(|id| {
        node_index(recovery.topology, id).map(|index| {
            recovery.topology.nodes()[index]
                .coordinates_m
                .canonical_value()
                .map(f64::to_bits)
        })
    });
    tracing::error!(event_code="meshing.facet_support.no_ear",constraint_index,boundary_nodes=boundary.len(),
        truncated=boundary.len()>MAXIMUM_WITNESS_NODES,?axes,?sign,?source,?source_points,
        local_boundary=%serde_json::to_string(&nodes).expect("finite integer witness serialization"),
        "facet support has no admissible ear");
    Ok(())
}
