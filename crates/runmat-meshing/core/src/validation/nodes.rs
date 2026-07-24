use std::collections::BTreeSet;

use crate::contracts::AnalysisMeshArtifact;

use super::AnalysisMeshValidationError;

pub(super) fn validate_nodes(
    mesh: &AnalysisMeshArtifact,
) -> Result<BTreeSet<u32>, AnalysisMeshValidationError> {
    let mut node_ids = BTreeSet::<u32>::new();
    for node in &mesh.nodes {
        if !node_ids.insert(node.node_id) {
            return Err(AnalysisMeshValidationError::DuplicateNodeId {
                node_id: node.node_id,
            });
        }
        if node
            .coordinates_m
            .iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(AnalysisMeshValidationError::NonFiniteNodeCoordinate {
                node_id: node.node_id,
            });
        }
    }
    Ok(node_ids)
}
