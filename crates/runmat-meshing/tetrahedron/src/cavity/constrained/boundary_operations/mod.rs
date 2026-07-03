use super::*;

mod edge_patch;
pub use edge_patch::split_constrained_cavity_boundary_edge_patch_at_centroid;
#[cfg(test)]
pub(crate) use edge_patch::split_constrained_cavity_boundary_edge_patch_with_weights;
mod face_splits;
pub(super) use face_splits::*;
pub use face_splits::{
    split_constrained_cavity_boundary_face, split_constrained_cavity_boundary_face_at_barycentric,
    split_constrained_cavity_boundary_face_at_centroid, split_constrained_cavity_boundary_faces,
    split_constrained_cavity_boundary_faces_at_centroids,
};
mod source_edge;
pub use source_edge::split_constrained_cavity_source_edge;

pub fn split_constrained_cavity_boundary_edge(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge: [u32; 2],
) -> Result<(ConstrainedCavity, ConstrainedCavityNode), ConstrainedCavityBoundaryEdgeSplitError> {
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
    let Some(target_face) = cavity.boundary_faces.iter().find(|face| {
        face_edges(face.node_ids)
            .into_iter()
            .any(|candidate| sorted_edge(candidate) == target_edge)
    }) else {
        return Err(
            ConstrainedCavityBoundaryEdgeSplitError::MissingBoundaryEdge {
                node_ids: target_edge,
            },
        );
    };
    let split_node = boundary_edge_split_node(target_edge, &boundary_node_map, 0.5);
    let split_faces = split_constrained_cavity_boundary_faces_on_edge(
        &cavity.boundary_faces,
        target_face.node_ids,
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

pub fn split_constrained_cavity_boundary_patch_at_centroids(
    cavity: &ConstrainedCavity,
    boundary_nodes: &[ConstrainedCavityNode],
    edge_patches: &[[u32; 2]],
    faces: &[[u32; 3]],
) -> Result<ConstrainedCavityBoundaryPatchSplit, ConstrainedCavityBoundaryPatchSplitError> {
    let mut split_cavity = cavity.clone();
    let mut current_nodes = boundary_nodes.to_vec();
    let mut split_nodes = Vec::<ConstrainedCavityNode>::new();
    let mut steps = Vec::<ConstrainedCavityBoundaryPatchSplitStep>::new();

    for edge in edge_patches {
        let target_edge = sorted_edge(*edge);
        let (next_cavity, split_node) = split_constrained_cavity_boundary_edge_patch_at_centroid(
            &split_cavity,
            &current_nodes,
            target_edge,
        )
        .map_err(ConstrainedCavityBoundaryPatchSplitError::Edge)?;
        steps.push(ConstrainedCavityBoundaryPatchSplitStep::EdgePatch {
            node_ids: target_edge,
            split_node_id: split_node.node_id,
        });
        current_nodes.push(split_node.clone());
        split_nodes.push(split_node);
        split_cavity = next_cavity;
    }

    let mut seen_faces = BTreeSet::<[u32; 3]>::new();
    for face in faces {
        let target_face = sorted_face(*face);
        if !seen_faces.insert(target_face) {
            return Err(ConstrainedCavityBoundaryPatchSplitError::Face(
                ConstrainedCavityBoundaryFaceSplitError::DuplicateBoundaryFace {
                    node_ids: target_face,
                },
            ));
        }
        let (next_cavity, split_node) = split_constrained_cavity_boundary_face_at_centroid(
            &split_cavity,
            &current_nodes,
            target_face,
        )
        .map_err(ConstrainedCavityBoundaryPatchSplitError::Face)?;
        steps.push(ConstrainedCavityBoundaryPatchSplitStep::Face {
            node_ids: target_face,
            split_node_id: split_node.node_id,
        });
        current_nodes.push(split_node.clone());
        split_nodes.push(split_node);
        split_cavity = next_cavity;
    }

    Ok(ConstrainedCavityBoundaryPatchSplit {
        cavity: split_cavity,
        split_nodes,
        steps,
    })
}
