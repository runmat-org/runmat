use std::collections::BTreeSet;

use crate::artifact::AnalysisMeshArtifact;

use super::{connectivity::sorted_edge, AnalysisMeshValidationError};

pub(super) fn validate_boundary_edges(
    mesh: &AnalysisMeshArtifact,
    node_ids: &BTreeSet<u32>,
    face_ids: &BTreeSet<String>,
) -> Result<BTreeSet<[u32; 2]>, AnalysisMeshValidationError> {
    let mut boundary_edge_ids = BTreeSet::<String>::new();
    let mut recovered_boundary_edges = BTreeSet::<[u32; 2]>::new();
    for edge in &mesh.boundary_edges {
        if !boundary_edge_ids.insert(edge.edge_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateBoundaryEdgeId {
                edge_id: edge.edge_id.clone(),
            });
        }
        if edge.node_ids.len() != 2 {
            return Err(AnalysisMeshValidationError::WrongBoundaryEdgeNodeCount {
                edge_id: edge.edge_id.clone(),
                expected: 2,
                actual: edge.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &edge.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownBoundaryEdgeNode {
                    edge_id: edge.edge_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedBoundaryEdgeNode {
                    edge_id: edge.edge_id.clone(),
                });
            }
        }
        for face_id in &edge.adjacent_boundary_face_ids {
            if !face_ids.contains(face_id) {
                return Err(
                    AnalysisMeshValidationError::UnknownBoundaryEdgeAdjacentFace {
                        edge_id: edge.edge_id.clone(),
                        face_id: face_id.clone(),
                    },
                );
            }
        }
        if !edge.adjacent_boundary_face_ids.is_empty() {
            recovered_boundary_edges.insert(sorted_edge(edge.node_ids[0], edge.node_ids[1]));
        }
    }
    Ok(recovered_boundary_edges)
}
