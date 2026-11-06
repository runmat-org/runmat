use crate::{
    cavity::constrained::{
        ConstrainedCavity, ConstrainedCavityBoundaryFace, ConstrainedCavityNode,
        ConstrainedCavityRefillOptions,
    },
    cdt::{
        insertion::validate_constrained_delaunay_volume_topology,
        topology::build_delaunay_volume_topology_with_regions, DelaunayInsertionErrorKind,
        DelaunayTopologyErrorKind, DelaunayVolumeNode, DelaunayVolumeTopology,
    },
};
use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use super::{
    canonical_face, invalid_topology, recovered_segments_exist, refill_side_exact_with_options,
    resource_or_cancelled,
};
use crate::cdt::facet_recovery::{
    DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle,
    DelaunaySegmentRecovery, FacetRecoveryWork,
};

type RegionalTetrahedron = ([u32; 4], Option<PersistentEntityId>);
type PrunedCavity = (Vec<DelaunayVolumeNode>, Vec<RegionalTetrahedron>);

pub(super) struct CavityRefillRequest<'a> {
    pub(super) inserted_nodes: &'a [DelaunayVolumeNode],
    pub(super) facet: [StableDigest; 3],
    pub(super) protected_facets: &'a [DelaunayRecoveredFacetTriangle],
    pub(super) cavities: &'a [ConstrainedCavity],
    pub(super) refill_options: ConstrainedCavityRefillOptions,
    pub(super) constraint_index: u32,
}

/// Refills one or both unfillable sides from the same immutable cavity while admitting their
/// Steiner nodes as part of that same topology mutation. The nodes are not globally inserted
/// first: doing so could alter recovered constraints outside the cavity that owns them.
pub(super) fn refill_cavity_sides(
    original: &DelaunaySegmentRecovery,
    request: CavityRefillRequest<'_>,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayFacetRecoveryError> {
    let CavityRefillRequest {
        inserted_nodes,
        facet,
        protected_facets,
        cavities,
        refill_options,
        constraint_index,
    } = request;
    if cavities.is_empty()
        || cavities.len() > 2
        || cavities
            .windows(2)
            .any(|pair| pair[0].removed_tetrahedron_ids != pair[1].removed_tetrahedron_ids)
    {
        work.reject("atomic facet refill received inconsistent cavity sides");
        return Ok(None);
    }
    let Some((nodes, remap)) = canonical_node_inventory(original, inserted_nodes) else {
        work.reject("atomic facet refill received duplicate Steiner node evidence");
        return Ok(None);
    };
    let remapped = cavities
        .iter()
        .map(|cavity| remap_cavity(cavity, &remap, constraint_index))
        .collect::<Result<Vec<_>, _>>()?;
    let removed = cavities[0]
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut local_node_ids = removed
        .iter()
        .flat_map(|tetrahedron| {
            original.topology.tetrahedra[*tetrahedron as usize]
                .vertex_indices
                .map(|node| remap[node as usize])
        })
        .collect::<std::collections::BTreeSet<_>>();
    for inserted in inserted_nodes {
        if let Ok(index) = nodes.binary_search_by_key(&inserted.identity, |node| node.identity) {
            local_node_ids.insert(index as u32);
        }
    }
    let cavity_nodes = nodes
        .iter()
        .enumerate()
        .filter(|(node_id, _)| local_node_ids.contains(&(*node_id as u32)))
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(node.coordinates_m.canonical_value()),
        })
        .collect::<Vec<_>>();
    let mut refills = Vec::with_capacity(cavities.len());
    for cavity in &remapped {
        let Some(refill) = refill_side_exact_with_options(
            cavity,
            &cavity_nodes,
            refill_options,
            constraint_index,
            work,
        )?
        else {
            let detail = work
                .last_rejection()
                .unwrap_or("the cavity refiller returned no candidate");
            work.reject(format!(
                "atomic facet cavity side could not be retriangulated: {detail}"
            ));
            return Ok(None);
        };
        refills.extend(refill);
    }

    let mut tetrahedra = original
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| {
            Ok((
                remap_nodes(tetrahedron.vertex_indices, &remap, constraint_index)?,
                tetrahedron.region_id.clone(),
            ))
        })
        .collect::<Result<Vec<_>, DelaunayFacetRecoveryError>>()?;
    tetrahedra.extend(refills.into_iter().map(|tetrahedron| (tetrahedron, None)));
    let (nodes, tetrahedra) =
        prune_unreferenced_inserted_nodes(original, nodes, tetrahedra, constraint_index)?;
    let candidate = match build_delaunay_volume_topology_with_regions(
        nodes,
        tetrahedra,
        work.options.segment_recovery.insertion.topology,
        work.cancellation,
    ) {
        Ok(candidate) => candidate,
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            DelaunayTopologyErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            _ => {
                work.reject(format!(
                    "atomic facet cavity topology was invalid: {failure}"
                ));
                return Ok(None);
            }
        },
    };
    let mut protected = protected_facets
        .iter()
        .map(|triangle| canonical_face(triangle.node_identities))
        .collect::<Vec<_>>();
    protected.push(canonical_face(facet));
    protected.sort_unstable();
    protected.dedup();
    match validate_constrained_delaunay_volume_topology(
        &candidate,
        &protected,
        work.options.segment_recovery.insertion,
        work.cancellation,
    ) {
        Ok(()) => {}
        Err(failure) => match failure.kind {
            DelaunayInsertionErrorKind::ResourceLimit => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::ResourceLimit,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            DelaunayInsertionErrorKind::Cancelled => {
                return Err(resource_or_cancelled(
                    DelaunayFacetRecoveryErrorKind::Cancelled,
                    constraint_index,
                    failure.to_string(),
                ));
            }
            _ => {
                work.reject(format!(
                    "atomic facet cavity was not constrained-Delaunay: {failure}"
                ));
                return Ok(None);
            }
        },
    }
    let mut candidate_recovery = original.clone();
    candidate_recovery.topology = candidate.clone();
    if recovered_segments_exist(&candidate_recovery) {
        Ok(Some(candidate))
    } else {
        work.reject("atomic facet cavity removed a recovered segment");
        Ok(None)
    }
}

fn prune_unreferenced_inserted_nodes(
    original: &DelaunaySegmentRecovery,
    nodes: Vec<DelaunayVolumeNode>,
    tetrahedra: Vec<RegionalTetrahedron>,
    constraint_index: u32,
) -> Result<PrunedCavity, DelaunayFacetRecoveryError> {
    let original_identities = original
        .topology
        .nodes
        .iter()
        .map(|node| node.identity)
        .collect::<std::collections::BTreeSet<_>>();
    let referenced = tetrahedra
        .iter()
        .flat_map(|(tetrahedron, _)| *tetrahedron)
        .collect::<std::collections::BTreeSet<_>>();
    let mut remap = vec![None; nodes.len()];
    let mut retained = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.into_iter().enumerate() {
        if original_identities.contains(&node.identity) || referenced.contains(&(index as u32)) {
            remap[index] = Some(retained.len() as u32);
            retained.push(node);
        }
    }
    let tetrahedra = tetrahedra
        .into_iter()
        .map(|(tetrahedron, region)| {
            Ok((
                tetrahedron
                    .map(|node| {
                        remap.get(node as usize).copied().flatten().ok_or_else(|| {
                            invalid_topology(
                                constraint_index,
                                "atomic facet refill references a pruned or out-of-bounds node",
                            )
                        })
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| {
                        invalid_topology(
                            constraint_index,
                            "atomic facet refill produced invalid tetrahedron arity",
                        )
                    })?,
                region,
            ))
        })
        .collect::<Result<Vec<_>, DelaunayFacetRecoveryError>>()?;
    Ok((retained, tetrahedra))
}

fn remap_cavity(
    cavity: &ConstrainedCavity,
    remap: &[u32],
    constraint_index: u32,
) -> Result<ConstrainedCavity, DelaunayFacetRecoveryError> {
    Ok(ConstrainedCavity {
        removed_tetrahedron_ids: cavity.removed_tetrahedron_ids.clone(),
        boundary_faces: cavity
            .boundary_faces
            .iter()
            .map(|face| {
                Ok(ConstrainedCavityBoundaryFace {
                    node_ids: remap_nodes(face.node_ids, remap, constraint_index)?,
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id: face.source_face_id,
                    source_edge_ids: face.source_edge_ids,
                    region_ids: face.region_ids.clone(),
                })
            })
            .collect::<Result<Vec<_>, DelaunayFacetRecoveryError>>()?,
        protected_node_ids: cavity
            .protected_node_ids
            .iter()
            .map(|node| remap_node(*node, remap, constraint_index))
            .collect::<Result<Vec<_>, _>>()?,
        target_volume_m3: cavity.target_volume_m3,
    })
}

fn remap_nodes<const N: usize>(
    nodes: [u32; N],
    remap: &[u32],
    constraint_index: u32,
) -> Result<[u32; N], DelaunayFacetRecoveryError> {
    nodes
        .map(|node| remap_node(node, remap, constraint_index))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map(|nodes| std::array::from_fn(|index| nodes[index]))
}

fn remap_node(
    node: u32,
    remap: &[u32],
    constraint_index: u32,
) -> Result<u32, DelaunayFacetRecoveryError> {
    remap
        .get(node as usize)
        .copied()
        .ok_or_else(|| invalid_topology(constraint_index, "facet cavity node is out of bounds"))
}

fn canonical_node_inventory(
    original: &DelaunaySegmentRecovery,
    inserted: &[DelaunayVolumeNode],
) -> Option<(Vec<DelaunayVolumeNode>, Vec<u32>)> {
    let mut nodes = original.topology.nodes.clone();
    nodes.extend_from_slice(inserted);
    nodes.sort_by_key(|node| node.identity);
    // Exact coordinate equality (with 0.0 == -0.0) through one ordered set, not a pairwise scan.
    let mut positions = std::collections::BTreeSet::new();
    if nodes
        .windows(2)
        .any(|pair| pair[0].identity == pair[1].identity)
        || !nodes.iter().all(|node| {
            positions.insert(
                node.coordinates_m
                    .canonical_value()
                    .map(|value| (value + 0.0).to_bits()),
            )
        })
    {
        return None;
    }
    let mut remap = Vec::with_capacity(original.topology.nodes.len());
    for node in &original.topology.nodes {
        remap.push(
            nodes
                .binary_search_by_key(&node.identity, |candidate| candidate.identity)
                .ok()? as u32,
        );
    }
    Some((nodes, remap))
}
