use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use crate::cdt::{
    topology::build_delaunay_volume_topology_with_regions, DelaunayConstraintFacet,
    DelaunayConstraintFacetSide, DelaunayConstraintNode, DelaunayConstraintSegment,
    DelaunayConstraints, DelaunayFacetSteinerInsertion, DelaunayTopologyOptions,
    DelaunayVolumeMesh, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind,
    DelaunayVolumeMeshStage, DelaunayVolumeNode, DelaunayVolumeProvenance,
    DelaunayVolumeSliverMutation,
};

pub(super) struct ConstraintComponent {
    pub region: PersistentEntityId,
    pub constraints: DelaunayConstraints,
    pub global_facet_indices: Vec<u32>,
}

pub(super) fn partition_constraints(
    constraints: &DelaunayConstraints,
) -> Result<Vec<ConstraintComponent>, DelaunayVolumeMeshError> {
    let region_components = regions(&constraints.facets)
        .into_iter()
        .map(|region| BTreeSet::from([region]))
        .collect::<Vec<_>>();
    if region_components.is_empty() {
        return Err(super::error(
            DelaunayVolumeMeshStage::Constraints,
            DelaunayVolumeMeshErrorKind::InvalidGeometry,
            "volume constraints contain no persistent region",
        ));
    }
    region_components
        .iter()
        .map(|regions| build_component(constraints, regions))
        .collect()
}

pub(super) fn remap_facet_insertions(
    insertions: &mut [DelaunayFacetSteinerInsertion],
    global_facet_indices: &[u32],
) -> Result<(), DelaunayVolumeMeshError> {
    for insertion in insertions {
        insertion.constraint_index = *global_facet_indices
            .get(insertion.constraint_index as usize)
            .ok_or_else(|| {
                super::error(
                    DelaunayVolumeMeshStage::FacetRecovery,
                    DelaunayVolumeMeshErrorKind::InvalidTopology,
                    "component facet insertion references an absent local constraint",
                )
            })?;
    }
    Ok(())
}

pub(super) fn assemble_topology(
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
                    return Err(super::error(
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
        super::error(
            DelaunayVolumeMeshStage::Refinement,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            format!("independent region assembly failed: {failure}"),
        )
    })
}

pub(super) fn assemble_provenance(meshes: &[DelaunayVolumeMesh]) -> DelaunayVolumeProvenance {
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

pub(super) fn assemble_insertions(
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

pub(super) fn assemble_refinement_insertions(
    meshes: &[DelaunayVolumeMesh],
) -> Vec<DelaunayVolumeNode> {
    meshes
        .iter()
        .flat_map(|mesh| mesh.refinement_insertions.iter().copied())
        .collect()
}

pub(super) fn assemble_sliver_mutations(
    meshes: &[DelaunayVolumeMesh],
) -> Vec<DelaunayVolumeSliverMutation> {
    meshes
        .iter()
        .flat_map(|mesh| mesh.sliver_mutations.iter().cloned())
        .collect()
}

fn regions(facets: &[DelaunayConstraintFacet]) -> BTreeSet<PersistentEntityId> {
    let mut regions = BTreeSet::new();
    for facet in facets {
        regions.extend(facet_regions(facet));
    }
    regions
}

fn build_component(
    constraints: &DelaunayConstraints,
    regions: &BTreeSet<PersistentEntityId>,
) -> Result<ConstraintComponent, DelaunayVolumeMeshError> {
    let selected_facets = constraints
        .facets
        .iter()
        .enumerate()
        .filter(|(_, facet)| {
            facet_regions(facet)
                .iter()
                .any(|region| regions.contains(region))
        })
        .collect::<Vec<_>>();
    let referenced_indices = selected_facets
        .iter()
        .flat_map(|(_, facet)| facet.vertex_indices)
        .collect::<BTreeSet<_>>();
    let nodes = referenced_indices
        .iter()
        .map(|index| constraints.nodes[*index as usize].clone())
        .collect::<Vec<DelaunayConstraintNode>>();
    let new_index_by_old = referenced_indices
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new as u32))
        .collect::<BTreeMap<_, _>>();
    let facets = selected_facets
        .iter()
        .map(|(_, facet)| DelaunayConstraintFacet {
            vertex_indices: facet.vertex_indices.map(|index| new_index_by_old[&index]),
            positive_side: localize_side(&facet.positive_side, regions),
            negative_side: localize_side(&facet.negative_side, regions),
            ..(*facet).clone()
        })
        .collect::<Vec<_>>();
    let facet_segment_keys = facets
        .iter()
        .flat_map(|facet| {
            (0..3).map(|edge| {
                sorted_segment([
                    facet.vertex_indices[edge],
                    facet.vertex_indices[(edge + 1) % 3],
                ])
            })
        })
        .collect::<BTreeSet<_>>();
    let segments_by_identity = constraints
        .segments
        .iter()
        .map(|segment| {
            let mut identities = segment
                .vertex_indices
                .map(|index| constraints.nodes[index as usize].identity);
            identities.sort_unstable();
            (identities, segment)
        })
        .collect::<BTreeMap<_, _>>();
    let segments = facet_segment_keys
        .into_iter()
        .map(|indices| {
            let identities = indices.map(|index| nodes[index as usize].identity);
            let source = segments_by_identity.get(&identities).ok_or_else(|| {
                super::error(
                    DelaunayVolumeMeshStage::Constraints,
                    DelaunayVolumeMeshErrorKind::InvalidTopology,
                    "component facet edge is absent from the full constraint inventory",
                )
            })?;
            Ok(DelaunayConstraintSegment {
                vertex_indices: indices,
                source_edge_id: source.source_edge_id.clone(),
                source_edge_parameters: source.source_edge_parameters,
            })
        })
        .collect::<Result<Vec<_>, DelaunayVolumeMeshError>>()?;
    Ok(ConstraintComponent {
        region: regions
            .first()
            .expect("region component is nonempty")
            .clone(),
        constraints: DelaunayConstraints {
            nodes,
            segments,
            facets,
        },
        global_facet_indices: selected_facets
            .iter()
            .map(|(index, _)| *index as u32)
            .collect(),
    })
}

fn localize_side(
    side: &DelaunayConstraintFacetSide,
    regions: &BTreeSet<PersistentEntityId>,
) -> DelaunayConstraintFacetSide {
    match side {
        DelaunayConstraintFacetSide::Region(region) if regions.contains(region) => {
            DelaunayConstraintFacetSide::Region(region.clone())
        }
        DelaunayConstraintFacetSide::Region(_)
        | DelaunayConstraintFacetSide::Exterior
        | DelaunayConstraintFacetSide::Void => DelaunayConstraintFacetSide::Exterior,
    }
}

fn facet_regions(facet: &DelaunayConstraintFacet) -> Vec<PersistentEntityId> {
    let mut regions = [&facet.positive_side, &facet.negative_side]
        .into_iter()
        .filter_map(|side| match side {
            DelaunayConstraintFacetSide::Region(region) => Some(region.clone()),
            DelaunayConstraintFacetSide::Exterior | DelaunayConstraintFacetSide::Void => None,
        })
        .collect::<Vec<_>>();
    regions.sort();
    regions.dedup();
    regions
}

fn sorted_segment(mut vertices: [u32; 2]) -> [u32; 2] {
    vertices.sort_unstable();
    vertices
}
