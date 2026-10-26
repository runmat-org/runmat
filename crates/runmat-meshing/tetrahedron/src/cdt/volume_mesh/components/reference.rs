use super::*;
use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions,
    DelaunayVolumeNode,
};
use crate::cdt::{DelaunayVolumeMesh, DelaunayVolumeProvenance};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
pub(in super::super::super) fn assemble_topology(
    meshes: &[DelaunayVolumeMesh],
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<crate::cdt::DelaunayVolumeTopology, DelaunayVolumeMeshError> {
    let mut nodes = BTreeMap::<StableDigest, DelaunayVolumeNode>::new();
    let mut tetrahedra = Vec::new();
    for mesh in meshes {
        for node in &mesh.topology.nodes {
            if let Some(existing) = nodes.insert(node.identity, *node) {
                if existing != *node {
                    return Err(super::super::error(
                        DelaunayVolumeMeshStage::Refinement,
                        DelaunayVolumeMeshErrorKind::InvalidTopology,
                        "independent regions produced conflicting coordinates for one node identity",
                    ));
                }
            }
        }
        for tetrahedron in &mesh.topology.tetrahedra {
            tetrahedra.push((
                tetrahedron
                    .vertex_indices
                    .map(|index| mesh.topology.nodes[index as usize].identity),
                tetrahedron.region_id.clone(),
            ));
        }
    }
    let nodes = nodes.into_values().collect::<Vec<_>>();
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index as u32))
        .collect::<BTreeMap<_, _>>();
    build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra
            .into_iter()
            .map(|(identities, region)| {
                (identities.map(|identity| node_indices[&identity]), region)
            })
            .collect(),
        options,
        cancellation,
    )
    .map_err(|failure| {
        super::super::error(
            DelaunayVolumeMeshStage::Refinement,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            format!("independent region assembly failed: {failure}"),
        )
    })
}

pub(in super::super::super) fn assemble_provenance(
    meshes: &[DelaunayVolumeMesh],
) -> DelaunayVolumeProvenance {
    let mut provenance = DelaunayVolumeProvenance {
        nodes: meshes
            .iter()
            .flat_map(|mesh| mesh.provenance.nodes.iter().cloned())
            .collect(),
        segments: meshes
            .iter()
            .flat_map(|mesh| mesh.provenance.segments.iter().cloned())
            .collect(),
        facets: meshes
            .iter()
            .flat_map(|mesh| mesh.provenance.facets.iter().cloned())
            .collect(),
    };
    provenance
        .nodes
        .sort_by_key(|binding| binding.node_identity);
    provenance.nodes =
        provenance
            .nodes
            .into_iter()
            .fold(Vec::new(), |mut merged, binding| {
                if let Some(previous) = merged.last_mut().filter(
                    |previous: &&mut crate::cdt::DelaunayNodeProvenance| {
                        previous.node_identity == binding.node_identity
                    },
                ) {
                    previous.entity_ids.extend(binding.entity_ids);
                    previous.entity_ids.sort();
                    previous.entity_ids.dedup();
                } else {
                    merged.push(binding);
                }
                merged
            });
    provenance
        .segments
        .sort_by_key(|binding| binding.node_identities);
    provenance.segments.dedup();
    provenance
        .facets
        .sort_by_key(|binding| binding.node_identities);
    provenance.facets =
        provenance
            .facets
            .into_iter()
            .fold(Vec::new(), |mut merged, binding| {
                if let Some(previous) = merged.last_mut().filter(
                    |previous: &&mut crate::cdt::DelaunayFacetProvenance| {
                        previous.node_identities == binding.node_identities
                            && previous.chart_id == binding.chart_id
                            && previous.entity_ids == binding.entity_ids
                    },
                ) {
                    previous.region_ids.extend(binding.region_ids);
                    previous.region_ids.sort();
                    previous.region_ids.dedup();
                } else {
                    merged.push(binding);
                }
                merged
            });
    provenance
}

pub(in super::super::super) fn assemble_insertions(
    meshes: &[DelaunayVolumeMesh],
) -> Vec<DelaunayFacetSteinerInsertion> {
    let mut insertions = meshes
        .iter()
        .flat_map(|mesh| mesh.facet_recovery_insertions.iter().copied())
        .collect::<Vec<_>>();
    insertions.sort_by_key(|insertion| {
        (
            insertion.constraint_index,
            insertion.support_node_identities,
            insertion.insertion_round,
            insertion.candidate_rank,
            insertion.node_identity,
        )
    });
    insertions
}

pub(in super::super::super) fn assemble_quality_mutations(
    meshes: &[DelaunayVolumeMesh],
) -> Vec<crate::cdt::DelaunayVolumeQualityMutation> {
    meshes
        .iter()
        .flat_map(|mesh| mesh.quality_mutations.iter().cloned())
        .collect()
}
