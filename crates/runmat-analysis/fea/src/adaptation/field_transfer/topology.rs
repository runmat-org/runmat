use std::collections::BTreeMap;

use runmat_analysis_core::AnalysisField;
use runmat_meshing_core::{
    ElementOrder, FieldTopologyLocation, SolverEntityTransfer, SolverMeshArtifact,
    SolverMeshTransferMap, StableDigest, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};

use super::SolverFieldTransferError;

pub(super) fn transfer_map_for_location(
    transfer_map: &SolverMeshTransferMap,
    location: FieldTopologyLocation,
) -> BTreeMap<StableDigest, &SolverEntityTransfer> {
    let transfers = match location {
        FieldTopologyLocation::Node => &transfer_map.node_transfers,
        FieldTopologyLocation::VolumeElement => &transfer_map.volume_element_transfers,
        FieldTopologyLocation::BoundaryFace => &transfer_map.boundary_face_transfers,
        FieldTopologyLocation::BoundaryEdge => &transfer_map.boundary_edge_transfers,
    };
    transfers
        .iter()
        .map(|transfer| (transfer.target_stable_identity, transfer))
        .collect()
}

pub(super) fn stable_identities(
    artifact: &SolverMeshArtifact,
    location: FieldTopologyLocation,
) -> BTreeMap<u64, StableDigest> {
    match location {
        FieldTopologyLocation::Node => artifact
            .topology
            .nodes
            .iter()
            .map(|entity| (entity.node_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::VolumeElement => artifact
            .topology
            .volume_elements
            .iter()
            .map(|entity| (entity.element_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::BoundaryFace => artifact
            .topology
            .boundary_faces
            .iter()
            .map(|entity| (entity.face_id, entity.stable_identity))
            .collect(),
        FieldTopologyLocation::BoundaryEdge => artifact
            .topology
            .boundary_edges
            .iter()
            .map(|entity| (entity.edge_id, entity.stable_identity))
            .collect(),
    }
}

pub(super) fn component_count(
    field: &AnalysisField,
    entity_count: usize,
    value_count: usize,
) -> Result<usize, SolverFieldTransferError> {
    if field.shape.is_empty() || field.shape[0] != entity_count {
        return Err(SolverFieldTransferError::InvalidFieldShape);
    }
    let components = field.shape[1..]
        .iter()
        .try_fold(1_usize, |product, count| product.checked_mul(*count))
        .ok_or(SolverFieldTransferError::InvalidFieldShape)?;
    if components == 0
        || entity_count
            .checked_mul(components)
            .is_none_or(|expected| expected != value_count)
    {
        return Err(SolverFieldTransferError::InvalidFieldShape);
    }
    Ok(components)
}

pub(super) fn target_midpoint_edges(
    target: &SolverMeshArtifact,
) -> Result<BTreeMap<StableDigest, [StableDigest; 2]>, SolverFieldTransferError> {
    if target.resolved_request.element_order == ElementOrder::Tet4 {
        return Ok(BTreeMap::new());
    }
    let mut result = BTreeMap::new();
    let identities = stable_identities(target, FieldTopologyLocation::Node);
    for element in &target.topology.volume_elements {
        for (local_edge, corners) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.iter().enumerate() {
            let mut edge = [
                identities[&element.node_ids[corners[0]]],
                identities[&element.node_ids[corners[1]]],
            ];
            edge.sort_unstable();
            let midpoint = identities[&element.node_ids[4 + local_edge]];
            if result
                .insert(midpoint, edge)
                .is_some_and(|existing| existing != edge)
            {
                return Err(SolverFieldTransferError::InconsistentQuadraticConnectivity);
            }
        }
    }
    Ok(result)
}
