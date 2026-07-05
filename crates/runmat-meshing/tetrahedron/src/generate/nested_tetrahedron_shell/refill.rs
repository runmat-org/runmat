use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{ProtectedBoundaryComplex, TopologyEntityId};

use crate::cavity::constrained::{
    retriangulate_constrained_cavity_from_nodes, ConstrainedCavity, ConstrainedCavityBoundaryFace,
    ConstrainedCavityNode, ConstrainedCavityRefill, ConstrainedCavityRefillOptions,
};
use crate::generate::TetrahedronGenerationError;

pub(super) struct NestedTetrahedronShellRefill {
    pub(super) cavity_id_to_node_id: BTreeMap<u32, TopologyEntityId>,
    pub(super) refill: ConstrainedCavityRefill,
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
        refill,
    })
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
