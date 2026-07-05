use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{MeshingStage, ProtectedBoundaryComplex, TopologyEntityId};

use crate::cavity::constrained::{
    retriangulate_constrained_cavity_from_nodes,
    split_constrained_cavity_boundary_faces_at_centroids, ConstrainedCavity,
    ConstrainedCavityBoundaryFace, ConstrainedCavityNode, ConstrainedCavityRefill,
    ConstrainedCavityRefillOptions,
};
use crate::generate::TetrahedronGenerationError;

pub(super) struct NestedTetrahedronShellRefill {
    pub(super) cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
    pub(super) split_nodes: Vec<ConstrainedCavityNode>,
    pub(super) strategy: NestedTetrahedronShellRefillStrategy,
    pub(super) boundary_centroid_refinement_rejected: bool,
    pub(super) refill: ConstrainedCavityRefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NestedTetrahedronShellRefillStrategy {
    BoundaryCentroidRefinement,
    BoundaryExactCover,
}

pub(super) fn refill_nested_tetrahedron_shell_cavity(
    plc: &ProtectedBoundaryComplex,
    target_volume_m3: f64,
) -> Result<NestedTetrahedronShellRefill, TetrahedronGenerationError> {
    let mut node_id_to_cavity_id = BTreeMap::<TopologyEntityId, u32>::new();
    let mut cavity_id_to_node_id = BTreeMap::<u32, TopologyEntityId>::new();
    let mut cavity_nodes = Vec::<ConstrainedCavityNode>::with_capacity(plc.nodes.len());
    for (index, node) in plc.nodes.iter().enumerate() {
        let cavity_id = u32::try_from(index)
            .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;
        node_id_to_cavity_id.insert(node.node_id.clone(), cavity_id);
        cavity_id_to_node_id.insert(cavity_id, node.node_id.clone());
        cavity_nodes.push(ConstrainedCavityNode {
            node_id: cavity_id,
            coordinates_m: node.coordinates_m,
        });
    }

    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: plc
            .facets
            .iter()
            .enumerate()
            .map(|(index, facet)| {
                let source_face_id = u32::try_from(index).ok();
                Ok(ConstrainedCavityBoundaryFace {
                    node_ids: [
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[0])?,
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[1])?,
                        cavity_node_id(&node_id_to_cavity_id, &facet.node_ids[2])?,
                    ],
                    outside_tetrahedron_ids: Vec::new(),
                    source_face_id,
                    source_edge_ids: [None, None, None],
                    region_ids: facet.material_interface_ids.clone(),
                })
            })
            .collect::<Result<Vec<_>, TetrahedronGenerationError>>()?,
        protected_node_ids: Vec::new(),
        target_volume_m3,
    };

    if let Some(refined) =
        try_boundary_centroid_refinement(&cavity, &cavity_nodes, &cavity_id_to_node_id)?
    {
        return Ok(refined);
    }

    let refill_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 1.0e-12,
        ..ConstrainedCavityRefillOptions::default()
    };
    let refill =
        retriangulate_constrained_cavity_from_nodes(&cavity, &cavity_nodes, refill_options)
            .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?
            .ok_or(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;

    Ok(NestedTetrahedronShellRefill {
        cavity_id_to_node_id,
        split_nodes: Vec::new(),
        strategy: NestedTetrahedronShellRefillStrategy::BoundaryExactCover,
        boundary_centroid_refinement_rejected: true,
        refill,
    })
}

fn try_boundary_centroid_refinement(
    cavity: &ConstrainedCavity,
    cavity_nodes: &[ConstrainedCavityNode],
    cavity_id_to_node_id: &BTreeMap<u32, TopologyEntityId>,
) -> Result<Option<NestedTetrahedronShellRefill>, TetrahedronGenerationError> {
    let split_faces = cavity
        .boundary_faces
        .iter()
        .map(|face| face.node_ids)
        .collect::<Vec<_>>();
    let Ok((cavity, split_nodes)) =
        split_constrained_cavity_boundary_faces_at_centroids(cavity, cavity_nodes, &split_faces)
    else {
        return Ok(None);
    };
    let mut cavity_id_to_node_id = cavity_id_to_node_id.clone();
    let mut cavity_nodes = cavity_nodes.to_vec();
    for split_node in &split_nodes {
        let node_id = TopologyEntityId {
            stage: MeshingStage::TetrahedronMesh,
            id: format!(
                "nested_tetrahedron_shell_boundary_node_{}",
                split_node.node_id
            ),
        };
        cavity_id_to_node_id.insert(split_node.node_id, node_id);
        cavity_nodes.push(split_node.clone());
    }
    let refill_options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.15,
        ..ConstrainedCavityRefillOptions::default()
    };
    let refill =
        retriangulate_constrained_cavity_from_nodes(&cavity, &cavity_nodes, refill_options)
            .map_err(|_| TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc)?;

    Ok(refill.map(|refill| NestedTetrahedronShellRefill {
        cavity_id_to_node_id,
        split_nodes,
        strategy: NestedTetrahedronShellRefillStrategy::BoundaryCentroidRefinement,
        boundary_centroid_refinement_rejected: false,
        refill,
    }))
}

fn cavity_node_id(
    node_id_to_cavity_id: &BTreeMap<TopologyEntityId, u32>,
    node_id: &TopologyEntityId,
) -> Result<u32, TetrahedronGenerationError> {
    node_id_to_cavity_id.get(node_id).copied().ok_or_else(|| {
        TetrahedronGenerationError::MissingPlcNode {
            node_id: node_id.id.clone(),
        }
    })
}
