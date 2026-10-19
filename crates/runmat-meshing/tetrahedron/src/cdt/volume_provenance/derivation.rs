use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

#[cfg(test)]
use super::validate_delaunay_volume_provenance;
use super::{
    checkpoint, error, DelaunayVolumeMetricContext, DelaunayVolumeProvenance,
    DelaunayVolumeProvenanceError, DelaunayVolumeProvenanceErrorKind,
    DelaunayVolumeProvenanceOptions, DelaunayVolumeTopology,
};

#[cfg(test)]
pub fn derive_delaunay_volume_metric_contexts(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<DelaunayVolumeMetricContext>, DelaunayVolumeProvenanceError> {
    validate_delaunay_volume_provenance(topology, provenance, options, cancellation)?;
    derive_delaunay_volume_metric_contexts_from_admitted_input(
        topology,
        provenance,
        options,
        cancellation,
    )
}

/// Derives per-cell metric context after the caller has admitted the topology and
/// persistent boundary provenance. Internal topology mutations use this path so
/// they do not revalidate immutable provenance for every speculative candidate.
pub(in crate::cdt) fn derive_delaunay_volume_metric_contexts_from_admitted_input(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<DelaunayVolumeMetricContext>, DelaunayVolumeProvenanceError> {
    let mut contexts = Vec::with_capacity(topology.tetrahedra.len());
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        contexts.push(derive_metric_context_for_cell(
            tetrahedron,
            topology,
            provenance,
        )?);
    }
    Ok(contexts)
}

/// Uses the admitted canonical provenance arrays directly. A topology mutation
/// needs context only for its changed cells, without rebuilding three maps.
pub(in crate::cdt) fn derive_metric_context_for_cell(
    tetrahedron: &super::super::DelaunayVolumeTetrahedron,
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
) -> Result<DelaunayVolumeMetricContext, DelaunayVolumeProvenanceError> {
    let identities = tetrahedron
        .vertex_indices
        .map(|vertex| topology.nodes[vertex as usize].identity);
    let region = tetrahedron.region_id.clone().ok_or_else(|| {
        error(
            DelaunayVolumeProvenanceErrorKind::InvalidTopology,
            "metric-context derivation requires assigned regions",
        )
    })?;
    Ok(derive_metric_context_for_corners(
        identities, region, provenance,
    ))
}

/// Shared identity-based derivation for a locally checked working cell.
pub(in crate::cdt) fn derive_metric_context_for_corners(
    identities: [StableDigest; 4],
    region: PersistentEntityId,
    provenance: &DelaunayVolumeProvenance,
) -> DelaunayVolumeMetricContext {
    let mut entities = BTreeSet::<PersistentEntityId>::from([region]);
    for identity in identities {
        extend(
            &mut entities,
            provenance
                .nodes
                .binary_search_by_key(&identity, |binding| binding.node_identity)
                .ok()
                .map(|index| provenance.nodes[index].entity_ids.as_slice()),
        );
    }
    for left in 0..4 {
        for right in (left + 1)..4 {
            let mut edge = [identities[left], identities[right]];
            edge.sort_unstable();
            extend(
                &mut entities,
                provenance
                    .segments
                    .binary_search_by_key(&edge, |binding| binding.node_identities)
                    .ok()
                    .map(|index| provenance.segments[index].entity_ids.as_slice()),
            );
        }
    }
    for opposite in 0..4 {
        let mut face = [StableDigest::ZERO; 3];
        let mut cursor = 0;
        for (vertex, identity) in identities.iter().enumerate() {
            if vertex != opposite {
                face[cursor] = *identity;
                cursor += 1;
            }
        }
        face.sort_unstable();
        extend(
            &mut entities,
            provenance
                .facets
                .binary_search_by_key(&face, |binding| binding.node_identities)
                .ok()
                .map(|index| provenance.facets[index].entity_ids.as_slice()),
        );
    }
    DelaunayVolumeMetricContext {
        tetrahedron_node_identities: identities,
        incident_entity_ids: entities.into_iter().collect(),
    }
}

fn extend(entities: &mut BTreeSet<PersistentEntityId>, binding: Option<&[PersistentEntityId]>) {
    if let Some(binding) = binding {
        entities.extend(binding.iter().cloned());
    }
}
