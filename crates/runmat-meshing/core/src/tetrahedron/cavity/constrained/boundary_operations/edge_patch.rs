use super::*;

pub fn split_constrained_cavity_boundary_edge_patch_at_centroid(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    split_constrained_cavity_boundary_edge_patch_with_weights_impl(
        cavity,
        boundary_nodes,
        edge,
        [0.25, 0.25, 0.25, 0.25],
    )
}

#[cfg(test)]
pub(crate) fn split_constrained_cavity_boundary_edge_patch_with_weights(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    weights: [f64; 4],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    split_constrained_cavity_boundary_edge_patch_with_weights_impl(
        cavity,
        boundary_nodes,
        edge,
        weights,
    )
}

pub(super) fn split_constrained_cavity_boundary_edge_patch_with_weights_impl(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
    weights: [f64; 4],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
    let weight_sum = weights.iter().sum::<f64>();
    let invalid_weights = weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || (weight_sum - 1.0).abs() > 1.0e-12;
    #[cfg(test)]
    {
        if invalid_weights {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::InvalidPatchWeights { weights });
        }
    }
    #[cfg(not(test))]
    debug_assert!(!invalid_weights);
    let target_edge = sorted_edge(edge);
    let boundary_node_map = boundary_nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for node_id in target_edge {
        if !boundary_node_map.contains_key(&node_id) {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryNode { node_id });
        }
    }
    let incident_faces = cavity
        .boundary_faces
        .iter()
        .filter(|face| {
            face_edges(face.node_ids)
                .into_iter()
                .any(|candidate| sorted_edge(candidate) == target_edge)
        })
        .collect::<Vec<_>>();
    if incident_faces.len() != 2 {
        return Err(
            ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryEdge {
                node_ids: target_edge,
            },
        );
    }
    let mut opposite_nodes = Vec::<u32>::new();
    for face in &incident_faces {
        let Some(opposite) = face
            .node_ids
            .into_iter()
            .find(|node_id| !target_edge.contains(node_id))
        else {
            return Err(ConstrainedCavityBoundaryEdgeSplitError::Split(
                ConstrainedCavityBoundarySplitError::MissingBoundaryFace {
                    node_ids: sorted_face(face.node_ids),
                },
            ));
        };
        if !boundary_node_map.contains_key(&opposite) {
            return Err(
                ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryNode { node_id: opposite },
            );
        }
        opposite_nodes.push(opposite);
    }
    let split_node = boundary_edge_patch_split_node(
        target_edge,
        [opposite_nodes[0], opposite_nodes[1]],
        &boundary_node_map,
        weights,
    );
    let split_faces = split_constrained_cavity_boundary_faces_on_edge_patch(
        &cavity.boundary_faces,
        target_edge,
        split_node.node_id,
    )
    .map_err(ConstrainedCavityBoundaryEdgeSplitError::Split)?;
    let mut split_cavity = cavity.clone();
    split_cavity.boundary_faces = split_faces;
    validate_constrained_cavity(&split_cavity)
        .map_err(ConstrainedCavityBoundaryEdgeSplitError::Validation)?;
    Ok((split_cavity, split_node))
}
