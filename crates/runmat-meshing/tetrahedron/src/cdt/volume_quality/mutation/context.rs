//! Borrow immutable provenance contexts across canonical cell mutations.
//! A region or oriented-identity change derives a fresh context. Metric grading
//! is deliberately not retained: current adjacency can change distant caps.
use super::*;

pub(super) enum MutationContext<'a> {
    Retained(&'a DelaunayTetrahedronQuality),
    Derived(DelaunayVolumeMetricContext),
}
impl MutationContext<'_> {
    pub(super) fn parts(&self) -> (&[StableDigest; 4], &[PersistentEntityId]) {
        match self {
            Self::Retained(old) => (&old.node_identities, &old.incident_metric_entity_ids),
            Self::Derived(context) => (
                &context.tetrahedron_node_identities,
                &context.incident_entity_ids,
            ),
        }
    }
}

pub(super) fn contexts<'a>(
    topology: &DelaunayVolumeTopology,
    baseline: &'a DelaunayVolumeQuality,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeQualityOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MutationContext<'a>>, DelaunayVolumeQualityError> {
    let mut old = baseline.tetrahedra.iter().peekable();
    topology
        .tetrahedra
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            checkpoint(index as u64, options, cancellation)?;
            let identities = cell
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            let current_key = key(identities);
            while old
                .peek()
                .is_some_and(|old| key(old.node_identities) < current_key)
            {
                old.next();
            }
            if let Some(old) = old.peek().filter(|old| {
                old.node_identities == identities && cell.region_id.as_ref() == Some(&old.region_id)
            }) {
                return Ok(MutationContext::Retained(old));
            }
            derive_metric_context_for_cell(cell, topology, provenance)
                .map(MutationContext::Derived)
                .map_err(provenance_error)
        })
        .collect()
}
