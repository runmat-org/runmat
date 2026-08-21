use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    checkpoint, error, validate_delaunay_volume_provenance, DelaunayVolumeMetricContext,
    DelaunayVolumeProvenance, DelaunayVolumeProvenanceError, DelaunayVolumeProvenanceErrorKind,
    DelaunayVolumeProvenanceOptions, DelaunayVolumeTopology,
};

pub fn derive_delaunay_volume_metric_contexts(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<DelaunayVolumeMetricContext>, DelaunayVolumeProvenanceError> {
    validate_delaunay_volume_provenance(topology, provenance, options, cancellation)?;
    let nodes = provenance
        .nodes
        .iter()
        .map(|binding| (binding.node_identity, binding.entity_ids.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let segments = provenance
        .segments
        .iter()
        .map(|binding| (binding.node_identities, binding.entity_ids.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let facets = provenance
        .facets
        .iter()
        .map(|binding| (binding.node_identities, binding.entity_ids.as_slice()))
        .collect::<BTreeMap<_, _>>();
    let mut contexts = Vec::with_capacity(topology.tetrahedra.len());
    for (index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        let identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        let region = tetrahedron.region_id.clone().ok_or_else(|| {
            error(
                DelaunayVolumeProvenanceErrorKind::InvalidTopology,
                "metric-context derivation requires assigned regions",
            )
        })?;
        let mut entities = BTreeSet::<PersistentEntityId>::from([region]);
        for identity in identities {
            extend(&mut entities, nodes.get(&identity));
        }
        for left in 0..4 {
            for right in (left + 1)..4 {
                let mut edge = [identities[left], identities[right]];
                edge.sort_unstable();
                extend(&mut entities, segments.get(&edge));
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
            extend(&mut entities, facets.get(&face));
        }
        contexts.push(DelaunayVolumeMetricContext {
            tetrahedron_node_identities: identities,
            incident_entity_ids: entities.into_iter().collect(),
        });
    }
    Ok(contexts)
}

fn extend(entities: &mut BTreeSet<PersistentEntityId>, binding: Option<&&[PersistentEntityId]>) {
    if let Some(binding) = binding {
        entities.extend(binding.iter().cloned());
    }
}
