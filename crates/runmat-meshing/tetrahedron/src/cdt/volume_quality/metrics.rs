//! Resolve metric evidence once per cell, using the topology's indexed support.
//! Construction and independent arithmetic validation share only grading policy.
use super::*;
use runmat_meshing_size::{
    grading::grade_indexed_metric_evaluations,
    metric::{ResolvedMetricEvaluation, ResolvedMetricField},
};

pub(super) fn resolve_graded_metrics(
    topology: &DelaunayVolumeTopology,
    contexts: &(impl MetricContextInventory + ?Sized),
    request: &MetricFieldRequest,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<CellMetricEvaluations, DelaunayVolumeQualityError> {
    checkpoint(0, options, cancellation)?;
    if contexts.len() != topology.tetrahedra.len() {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidMetricContext,
            None,
            "metric context inventory differs from the canonical cell inventory",
        ));
    }
    let field = ResolvedMetricField::new(request).map_err(metric_error)?;
    // Topology and context admission precede this internal helper. Constant
    // resolution needs neither a per-cell metric inventory nor grading scratch;
    // each consumer still checks every original cell and its metadata.
    if let Some(evaluation) = field.uniform_evaluation().map_err(metric_error)? {
        return Ok(CellMetricEvaluations::Uniform {
            evaluation,
            remaining: contexts.len(),
        });
    }
    Ok(CellMetricEvaluations::Indexed(
        resolve_indexed_metrics(
            topology,
            contexts.indexed().ok_or_else(|| {
                error(
                    DelaunayVolumeQualityErrorKind::InvalidMetricContext,
                    None,
                    "scoped grading requires complete metric contexts",
                )
            })?,
            &field,
            request,
            options,
            cancellation,
        )?
        .into_iter(),
    ))
}

fn resolve_indexed_metrics(
    topology: &DelaunayVolumeTopology,
    contexts: &[DelaunayVolumeMetricContext],
    field: &ResolvedMetricField,
    request: &MetricFieldRequest,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<ResolvedMetricEvaluation>, DelaunayVolumeQualityError> {
    resolve_context_metrics(
        topology,
        contexts.iter().map(|context| {
            (
                &context.tetrahedron_node_identities,
                context.incident_entity_ids.as_slice(),
            )
        }),
        field,
        request,
        options,
        cancellation,
    )
    .map(|(metrics, _)| metrics)
}

/// Resolve each distinct immutable entity set once, then grade the entire current
/// graph from raw values. Borrowed contexts do not copy retained entity strings.
pub(super) fn resolve_context_metrics<'a>(
    topology: &DelaunayVolumeTopology,
    contexts: impl ExactSizeIterator<Item = (&'a [StableDigest; 4], &'a [PersistentEntityId])>,
    field: &ResolvedMetricField,
    request: &MetricFieldRequest,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(Vec<ResolvedMetricEvaluation>, usize), DelaunayVolumeQualityError> {
    if contexts.len() != topology.tetrahedra.len() {
        return Err(error(
            DelaunayVolumeQualityErrorKind::InvalidMetricContext,
            None,
            "metric contexts must cover the complete current graph",
        ));
    }
    let mut metrics = Vec::with_capacity(contexts.len());
    let mut identities = Vec::with_capacity(contexts.len());
    let mut offsets = Vec::with_capacity(contexts.len() + 1);
    let mut neighbors = Vec::with_capacity(contexts.len().saturating_mul(4));
    let mut resolved = std::collections::BTreeMap::new();
    offsets.push(0);
    for (index, (cell, (identity, entities))) in
        topology.tetrahedra.iter().zip(contexts).enumerate()
    {
        checkpoint(index as u64, options, cancellation)?;
        let evaluation = match resolved.entry(entities) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let entities = entities.iter().cloned().collect::<BTreeSet<_>>();
                entry.insert(field.resolve(&entities).map_err(metric_error)?)
            }
        };
        metrics.push(evaluation.clone());
        identities.push(identity);
        neighbors.extend(
            cell.neighbors
                .iter()
                .flatten()
                .map(|neighbor| *neighbor as usize),
        );
        offsets.push(neighbors.len());
    }
    grade_indexed_metric_evaluations(
        request.maximum_grading_ratio,
        &identities,
        &offsets,
        &neighbors,
        &mut metrics,
    )
    .map_err(metric_error)?;
    Ok((metrics, resolved.len()))
}

pub(super) enum CellMetricEvaluations {
    Uniform {
        evaluation: ResolvedMetricEvaluation,
        remaining: usize,
    },
    Indexed(std::vec::IntoIter<ResolvedMetricEvaluation>),
}

impl Iterator for CellMetricEvaluations {
    type Item = ResolvedMetricEvaluation;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Uniform {
                evaluation,
                remaining,
            } if *remaining > 0 => {
                *remaining -= 1;
                Some(evaluation.clone())
            }
            Self::Uniform { .. } => None,
            Self::Indexed(values) => values.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = match self {
            Self::Uniform { remaining, .. } => *remaining,
            Self::Indexed(values) => values.len(),
        };
        (count, Some(count))
    }
}
impl ExactSizeIterator for CellMetricEvaluations {}

#[cfg(test)]
mod tests;
