//! Exact context and geometry reuse for admitted mutations. Scoped metrics are
//! regraded over the complete changed graph before any geometry is retained.
//! Connectivity alone never authorizes reuse after relocation or changed grading.
use super::super::volume_provenance::derive_metric_context_for_cell;
use super::*;
use runmat_meshing_size::metric::ResolvedMetricField;
mod context;

pub(in crate::cdt) fn evaluate_quality_after_mutation(
    topology: &DelaunayVolumeTopology,
    baseline: (&DelaunayVolumeTopology, &DelaunayVolumeQuality),
    metric_request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeQuality, DelaunayVolumeQualityError> {
    evaluate(
        topology,
        baseline,
        metric_request,
        provenance,
        options,
        cancellation,
    )
    .map(|(quality, _)| quality)
}

/// The baseline was admitted against this same immutable provenance and controls.
/// The candidate has already passed conforming-topology admission. This private
/// boundary cannot be used to accept decoded, unvalidated quality records.
fn evaluate(
    topology: &DelaunayVolumeTopology,
    (base, base_quality): (&DelaunayVolumeTopology, &DelaunayVolumeQuality),
    request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(DelaunayVolumeQuality, usize), DelaunayVolumeQualityError> {
    checkpoint(0, options, cancellation)?;
    if topology.nodes.len() as u64 > options.maximum_nodes
        || topology.tetrahedra.len() as u64 > options.maximum_tetrahedra
    {
        return Err(error(
            DelaunayVolumeQualityErrorKind::ResourceLimit,
            None,
            "volume topology exceeds the quality evaluation inventory limit",
        ));
    }
    let field = ResolvedMetricField::new(request).map_err(metric_error)?;
    let contexts = context::contexts(topology, base_quality, provenance, options, cancellation)?;
    let metrics = if let Some(evaluation) = field.uniform_evaluation().map_err(metric_error)? {
        metrics::CellMetricEvaluations::Uniform {
            evaluation,
            remaining: contexts.len(),
        }
    } else {
        let (metrics, _) = metrics::resolve_context_metrics(
            topology,
            contexts.iter().map(context::MutationContext::parts),
            &field,
            request,
            options,
            cancellation,
        )?;
        metrics::CellMetricEvaluations::Indexed(metrics.into_iter())
    };
    let unchanged_nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            checkpoint(index as u64, options, cancellation)?;
            Ok(base
                .nodes
                .binary_search_by_key(&node.identity, |old| old.identity)
                .is_ok_and(|index| {
                    base.nodes[index] == *node
                        && base.nodes[index]
                            .coordinates_m
                            .canonical_value()
                            .map(f64::to_bits)
                            == node.coordinates_m.canonical_value().map(f64::to_bits)
                }))
        })
        .collect::<Result<Vec<_>, DelaunayVolumeQualityError>>()?;
    let boundary_nodes = boundary_node_identities(provenance);
    let protected_facets = protected_facet_identities(provenance);
    let node_size = node_size::NodeSizeField::new(request, provenance)?;
    let mut cells = Vec::with_capacity(topology.tetrahedra.len());
    let mut baseline = base_quality.tetrahedra.iter().peekable();
    let mut reused = 0;
    for (index, ((cell, context), metric)) in topology
        .tetrahedra
        .iter()
        .zip(&contexts)
        .zip(metrics)
        .enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let identities = cell
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        let key = key(identities);
        while baseline
            .peek()
            .is_some_and(|old| self::key(old.node_identities) < key)
        {
            baseline.next();
        }
        let old = baseline
            .peek()
            .filter(|old| self::key(old.node_identities) == key);
        let region_id = cell.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeQualityErrorKind::InvalidTopology,
                Some(index),
                "every evaluated tetrahedron must have one assigned region",
            )
        })?;
        let corner_metrics = node_size.corner_metrics(
            &topology.nodes,
            cell.vertex_indices
                .map(|vertex| topology.nodes[vertex as usize]),
            &region_id,
        )?;
        if let Some(old) = old.filter(|old| {
            old.node_identities == identities
                && cell
                    .vertex_indices
                    .iter()
                    .all(|vertex| unchanged_nodes[*vertex as usize])
                && cell.region_id.as_ref() == Some(&old.region_id)
                && metric_bits(old.resolved_metric) == metric_bits(metric.metric)
                && old.active_metric_sources == metric.active_sources
                && old.applied_metric_contribution_count == metric.applied_contribution_count
                && old.clipped_metric_contribution_count == metric.clipped_contribution_count
                && old.rejected_metric_contribution_count == metric.rejected_contribution_count
        }) {
            let mut retained = (**old).clone();
            (
                retained.refinement_violation_ratio,
                retained.optimization_violation_ratio,
            ) = evaluation::violation_ratios(
                node_size::volume_owned_maximum_integrated_edge_length(
                    cell.vertex_indices.map(|vertex| {
                        topology.nodes[vertex as usize]
                            .coordinates_m
                            .canonical_value()
                    }),
                    corner_metrics,
                    protected_edge_mask(identities, &protected_facets),
                ),
                retained.metric_radius_edge_ratio,
                retained.metric_scaled_jacobian,
                retained.boundary_node_count,
                options,
            );
            cells.push(retained);
            reused += 1;
            continue;
        }
        cells.push(evaluate_tetrahedron(
            topology,
            index,
            evaluation::TetrahedronEvaluationContext {
                corner_metrics,
                region_id,
                boundary_node_count: identities
                    .iter()
                    .filter(|id| boundary_nodes.contains(id))
                    .count() as u8,
                protected_facet_count: protected_facet_count(identities, &protected_facets),
                protected_edges: protected_edge_mask(identities, &protected_facets),
                incident_metric_entity_ids: context.parts().1.to_vec(),
                metric: metric.clone(),
            },
            options,
        )?);
    }
    Ok((
        summary::summarize(topology, cells, options, cancellation)?,
        reused,
    ))
}

fn metric_bits(metric: MetricTensor3) -> [u64; 6] {
    [
        metric.xx, metric.yy, metric.zz, metric.xy, metric.xz, metric.yz,
    ]
    .map(f64::to_bits)
}

fn key(mut identities: [StableDigest; 4]) -> [StableDigest; 4] {
    identities.sort_unstable();
    identities
}

#[cfg(test)]
mod tests;
