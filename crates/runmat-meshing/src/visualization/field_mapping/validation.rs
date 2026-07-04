use std::collections::BTreeMap;

use runmat_meshing_core::contracts::AnalysisMeshArtifact;

use super::FieldMappingError;

pub(super) fn validate_nodal_vector_field<'a>(
    mesh: &'a AnalysisMeshArtifact,
    node_values: &'a [[f64; 3]],
) -> Result<BTreeMap<u32, [f64; 3]>, FieldMappingError> {
    if node_values.len() != mesh.nodes.len() {
        return Err(FieldMappingError::NodeVectorFieldLengthMismatch {
            node_value_count: node_values.len(),
            node_count: mesh.nodes.len(),
        });
    }
    for (node_index, value) in node_values.iter().enumerate() {
        for (component_index, component) in value.iter().enumerate() {
            if !component.is_finite() {
                return Err(FieldMappingError::NonFiniteNodeVectorValue {
                    node_index,
                    component_index,
                });
            }
        }
    }

    Ok(mesh
        .nodes
        .iter()
        .zip(node_values.iter().copied())
        .map(|(node, value)| (node.node_id, value))
        .collect())
}
