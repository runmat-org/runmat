use std::collections::BTreeSet;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    checkpoint, error, validate_options, DelaunayVolumeProvenance, DelaunayVolumeProvenanceError,
    DelaunayVolumeProvenanceErrorKind, DelaunayVolumeProvenanceOptions, DelaunayVolumeTopology,
};

pub fn validate_delaunay_volume_provenance(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    options: DelaunayVolumeProvenanceOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeProvenanceError> {
    validate_options(options)?;
    if provenance.nodes.len() as u64 > options.maximum_node_bindings
        || provenance.segments.len() as u64 > options.maximum_segment_bindings
        || provenance.facets.len() as u64 > options.maximum_facet_bindings
    {
        return Err(error(
            DelaunayVolumeProvenanceErrorKind::ResourceLimit,
            "provenance inventory exceeds its hard limit",
        ));
    }
    let node_identities = topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<BTreeSet<_>>();
    for (index, binding) in provenance.nodes.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        if binding.node_identity == StableDigest::ZERO
            || !node_identities.contains(&binding.node_identity)
            || index > 0 && provenance.nodes[index - 1].node_identity >= binding.node_identity
        {
            return Err(invalid(
                "node provenance must be canonically ordered and resolve to the topology",
            ));
        }
        validate_entities(&binding.entity_ids, &[PersistentEntityKind::Vertex])?;
    }
    for (index, binding) in provenance.segments.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        validate_simplex(binding.node_identities, &node_identities)?;
        if index > 0 && provenance.segments[index - 1].node_identities >= binding.node_identities {
            return Err(invalid("segment provenance must be canonically ordered"));
        }
        if !simplex_exists(topology, &binding.node_identities) {
            return Err(invalid("provenance segment is absent from the topology"));
        }
        validate_entities(&binding.entity_ids, &[PersistentEntityKind::Edge])?;
    }
    for (index, binding) in provenance.facets.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        validate_simplex(binding.node_identities, &node_identities)?;
        if index > 0 && provenance.facets[index - 1].node_identities >= binding.node_identities {
            return Err(invalid("facet provenance must be canonically ordered"));
        }
        validate_entities(
            &binding.entity_ids,
            &[PersistentEntityKind::Face, PersistentEntityKind::Contact],
        )?;
        validate_entities(&binding.region_ids, &[PersistentEntityKind::Region])?;
        let incident_regions = incident_regions(topology, binding.node_identities)?;
        if binding.region_ids != incident_regions {
            return Err(invalid(
                "facet provenance regions must exactly match topological incidence",
            ));
        }
    }
    Ok(())
}

fn validate_simplex<const N: usize>(
    identities: [StableDigest; N],
    topology_identities: &BTreeSet<StableDigest>,
) -> Result<(), DelaunayVolumeProvenanceError> {
    if identities.contains(&StableDigest::ZERO)
        || identities.windows(2).any(|pair| pair[0] >= pair[1])
        || identities
            .iter()
            .any(|identity| !topology_identities.contains(identity))
    {
        return Err(invalid(
            "provenance simplex identities must be ordered, unique, nonzero, and resolve",
        ));
    }
    Ok(())
}

fn validate_entities(
    entities: &[PersistentEntityId],
    allowed_kinds: &[PersistentEntityKind],
) -> Result<(), DelaunayVolumeProvenanceError> {
    if entities.is_empty() || entities.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            "persistent provenance entities must be nonempty, unique, and ordered",
        ));
    }
    for entity in entities {
        entity.validate().map_err(|failure| {
            invalid(format!(
                "persistent provenance entity is invalid: {failure}"
            ))
        })?;
        if !allowed_kinds.contains(&entity.kind) {
            return Err(invalid(
                "persistent provenance entity kind does not match its simplex dimension",
            ));
        }
    }
    Ok(())
}

fn simplex_exists<const N: usize>(
    topology: &DelaunayVolumeTopology,
    identities: &[StableDigest; N],
) -> bool {
    topology.tetrahedra.iter().any(|tetrahedron| {
        let tetrahedron_identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        identities
            .iter()
            .all(|identity| tetrahedron_identities.contains(identity))
    })
}

fn incident_regions(
    topology: &DelaunayVolumeTopology,
    identities: [StableDigest; 3],
) -> Result<Vec<PersistentEntityId>, DelaunayVolumeProvenanceError> {
    let mut regions = BTreeSet::new();
    for tetrahedron in &topology.tetrahedra {
        let tetrahedron_identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        if identities
            .iter()
            .all(|identity| tetrahedron_identities.contains(identity))
        {
            let region = tetrahedron.region_id.clone().ok_or_else(|| {
                error(
                    DelaunayVolumeProvenanceErrorKind::InvalidTopology,
                    "facet provenance requires assigned incident regions",
                )
            })?;
            regions.insert(region);
        }
    }
    if regions.is_empty() {
        return Err(invalid("provenance facet is absent from the topology"));
    }
    Ok(regions.into_iter().collect())
}

fn invalid(reason: impl Into<String>) -> DelaunayVolumeProvenanceError {
    error(DelaunayVolumeProvenanceErrorKind::InvalidProvenance, reason)
}
