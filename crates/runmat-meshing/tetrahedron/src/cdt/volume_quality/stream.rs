//! Shared cell reconstruction; callers collect or compare without a second inventory.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn visit_from_metric_contexts(
    topology: &DelaunayVolumeTopology,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeQualityOptions,
    metric_contexts: MetricContexts<'_>,
    boundary_nodes: &BTreeSet<StableDigest>,
    protected_facets: &ProtectedFacets,
    cancellation: &dyn MeshingCancellationSignal,
    mut visit: impl FnMut(usize, DelaunayTetrahedronQuality) -> Result<(), DelaunayVolumeQualityError>,
) -> Result<(), DelaunayVolumeQualityError> {
    let resolved_metrics = metrics::resolve_graded_metrics(
        topology,
        &metric_contexts,
        metric_request,
        options,
        cancellation,
    )?;

    for (index, ((tetrahedron, context), metric)) in topology
        .tetrahedra
        .iter()
        .zip(metric_contexts)
        .zip(resolved_metrics)
        .enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let context = context?;
        let region_id = tetrahedron.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                Some(index),
                "every evaluated tetrahedron must have one assigned region",
            )
        })?;
        let quality = evaluate_tetrahedron(
            topology,
            index,
            evaluation::TetrahedronEvaluationContext {
                region_id,
                incident_metric_entity_ids: context.incident_entity_ids,
                metric,
                boundary_node_count: context
                    .tetrahedron_node_identities
                    .iter()
                    .filter(|identity| boundary_nodes.contains(identity))
                    .count() as u8,
                protected_facet_count: protected_facet_count(
                    context.tetrahedron_node_identities,
                    protected_facets,
                ),
                protected_edges: super::protected_edge_mask(
                    context.tetrahedron_node_identities,
                    protected_facets,
                ),
            },
            options,
        )?;
        visit(index, quality)?;
    }

    Ok(())
}
