use crate::{
    cavity::constrained::{
        ConstrainedCavity, ConstrainedCavityBoundaryFace, ConstrainedCavityNode,
    },
    cdt::{
        insertion::validate_constrained_delaunay_volume_topology,
        topology::build_delaunay_volume_topology_with_regions, DelaunayInsertionErrorKind,
        DelaunayTopologyErrorKind, DelaunayVolumeTopology,
    },
};
use runmat_meshing_core::StableDigest;

use super::{
    canonical_face, invalid_topology, recovered_segments_exist, refill_side, resource_or_cancelled,
};
use crate::cdt::facet_recovery::{
    DelaunayFacetRecoveryError, DelaunayFacetRecoveryErrorKind, DelaunayRecoveredFacetTriangle,
    DelaunaySegmentRecovery, FacetRecoveryWork,
};

/// Refills both unfillable sides from the same immutable cavity after their independently admitted
/// Steiner nodes have been inserted. Reusing the immutable cavity avoids exposing a transient
/// pinched boundary between the two insertions as if it were a new geometric constraint.
pub(super) fn refill_simultaneous_sides(
    original: &DelaunaySegmentRecovery,
    inserted: &DelaunaySegmentRecovery,
    facet: [StableDigest; 3],
    protected_facets: &[DelaunayRecoveredFacetTriangle],
    cavities: &[ConstrainedCavity],
    constraint_index: u32,
    work: &mut FacetRecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayFacetRecoveryError> {
    if cavities.len() != 2
        || cavities[0].removed_tetrahedron_ids != cavities[1].removed_tetrahedron_ids
    {
        return Ok(None);
    }
    let remapped = cavities
        .iter()
        .map(|cavity| remap_cavity(cavity, original, inserted, constraint_index))
        .collect::<Result<Vec<_>, _>>()?;
    let nodes = inserted
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_id, node)| ConstrainedCavityNode {
            node_id: node_id as u32,
            coordinates_m: node.coordinates_m,
        })
        .collect::<Vec<_>>();
    let mut refills = Vec::with_capacity(2);
    for cavity in &remapped {
        let Some(refill) = refill_side(cavity, &nodes, &inserted.topology, constraint_index, work)?
        else {
            return Ok(None);
        };
        refills.extend(refill);
    }

    let removed = cavities[0]
        .removed_tetrahedron_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut tetrahedra = original
        .topology
        .tetrahedra
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(&(*index as u32)))
        .map(|(_, tetrahedron)| {
            Ok((
                remap_nodes(
                    tetrahedron.vertex_indices,
                    original,
                    inserted,
                    constraint_index,
                )?,
                tetrahedron.region_id.clone(),
            ))
        })
        .collect::<Result<Vec<_>, DelaunayFacetRecoveryError>>()?;
    tetrahedra.extend(refills.into_iter().map(|tetrahedron| (tetrahedron, None)));
    let candidate = match build_delaunay_volume_topology_with_regions(
        inserted.topology.nodes.clone(),
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
            _ => return Ok(None),
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
            _ => return Ok(None),
        },
    }
    let mut candidate_recovery = inserted.clone();
    candidate_recovery.topology = candidate.clone();
    Ok(recovered_segments_exist(&candidate_recovery).then_some(candidate))
}

fn remap_cavity(
    cavity: &ConstrainedCavity,
    original: &DelaunaySegmentRecovery,
    inserted: &DelaunaySegmentRecovery,
    constraint_index: u32,
) -> Result<ConstrainedCavity, DelaunayFacetRecoveryError> {
    Ok(ConstrainedCavity {
        removed_tetrahedron_ids: cavity.removed_tetrahedron_ids.clone(),
        boundary_faces: cavity
            .boundary_faces
            .iter()
            .map(|face| {
                Ok(ConstrainedCavityBoundaryFace {
                    node_ids: remap_nodes(face.node_ids, original, inserted, constraint_index)?,
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
            .map(|node| remap_node(*node, original, inserted, constraint_index))
            .collect::<Result<Vec<_>, _>>()?,
        target_volume_m3: cavity.target_volume_m3,
    })
}

fn remap_nodes<const N: usize>(
    nodes: [u32; N],
    original: &DelaunaySegmentRecovery,
    inserted: &DelaunaySegmentRecovery,
    constraint_index: u32,
) -> Result<[u32; N], DelaunayFacetRecoveryError> {
    nodes
        .map(|node| remap_node(node, original, inserted, constraint_index))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map(|nodes| std::array::from_fn(|index| nodes[index]))
}

fn remap_node(
    node: u32,
    original: &DelaunaySegmentRecovery,
    inserted: &DelaunaySegmentRecovery,
    constraint_index: u32,
) -> Result<u32, DelaunayFacetRecoveryError> {
    let identity = original
        .topology
        .nodes
        .get(node as usize)
        .ok_or_else(|| invalid_topology(constraint_index, "facet cavity node is out of bounds"))?
        .identity;
    inserted
        .topology
        .nodes
        .binary_search_by_key(&identity, |candidate| candidate.identity)
        .map(|index| index as u32)
        .map_err(|_| invalid_topology(constraint_index, "inserted topology lost a cavity node"))
}
