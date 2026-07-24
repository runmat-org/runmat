use std::collections::BTreeMap;

use runmat_meshing_core::contracts::AnalysisMeshArtifact;

mod error;
mod types;
mod validation;

pub use error::FieldMappingError;
pub use types::{BoundaryFaceScalarValue, BoundaryFaceVectorValue, BoundaryNodeVectorValue};

use validation::{
    validate_nodal_vector_field, validate_nodal_vector_to_boundary_face_topology,
    validate_nodal_vector_to_boundary_node_topology,
    validate_volume_scalar_to_boundary_face_topology,
};

pub fn map_volume_scalar_field_to_boundary_faces(
    mesh: &AnalysisMeshArtifact,
    element_values: &[f64],
) -> Result<Vec<BoundaryFaceScalarValue>, FieldMappingError> {
    validate_volume_scalar_to_boundary_face_topology(mesh)?;
    if element_values.len() != mesh.volume_elements.len() {
        return Err(FieldMappingError::ElementFieldLengthMismatch {
            element_value_count: element_values.len(),
            volume_element_count: mesh.volume_elements.len(),
        });
    }
    for (element_index, value) in element_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(FieldMappingError::NonFiniteElementValue { element_index });
        }
    }

    let element_values_by_id = mesh
        .volume_elements
        .iter()
        .zip(element_values.iter().copied())
        .map(|(element, value)| (element.element_id.as_str(), value))
        .collect::<BTreeMap<_, _>>();

    mesh.boundary_faces
        .iter()
        .map(|face| {
            if face.adjacent_volume_element_ids.is_empty() {
                return Err(FieldMappingError::BoundaryFaceMissingAdjacentVolume {
                    face_id: face.face_id.clone(),
                });
            }
            let mut value_sum = 0.0_f64;
            for volume_element_id in &face.adjacent_volume_element_ids {
                let Some(value) = element_values_by_id
                    .get(volume_element_id.as_str())
                    .copied()
                else {
                    return Err(FieldMappingError::BoundaryFaceReferencesUnknownVolume {
                        face_id: face.face_id.clone(),
                        volume_element_id: volume_element_id.clone(),
                    });
                };
                value_sum += value;
            }
            Ok(BoundaryFaceScalarValue {
                face_id: face.face_id.clone(),
                value: value_sum / face.adjacent_volume_element_ids.len() as f64,
            })
        })
        .collect()
}

pub fn map_nodal_vector_field_to_boundary_nodes(
    mesh: &AnalysisMeshArtifact,
    node_values: &[[f64; 3]],
) -> Result<Vec<BoundaryNodeVectorValue>, FieldMappingError> {
    validate_nodal_vector_to_boundary_node_topology(mesh)?;
    let node_values_by_id = validate_nodal_vector_field(mesh, node_values)?;
    let mut boundary_node_ids = BTreeMap::<u32, ()>::new();

    for face in &mesh.boundary_faces {
        for node_id in &face.node_ids {
            if !node_values_by_id.contains_key(node_id) {
                return Err(FieldMappingError::BoundaryFaceReferencesUnknownNode {
                    face_id: face.face_id.clone(),
                    node_id: *node_id,
                });
            }
            boundary_node_ids.insert(*node_id, ());
        }
    }
    for edge in &mesh.boundary_edges {
        for node_id in edge.node_ids {
            if !node_values_by_id.contains_key(&node_id) {
                return Err(FieldMappingError::BoundaryEdgeReferencesUnknownNode {
                    edge_id: edge.edge_id.clone(),
                    node_id,
                });
            }
            boundary_node_ids.insert(node_id, ());
        }
    }

    Ok(boundary_node_ids
        .keys()
        .map(|node_id| BoundaryNodeVectorValue {
            node_id: *node_id,
            value: node_values_by_id[node_id],
        })
        .collect())
}

pub fn map_nodal_vector_field_to_boundary_faces(
    mesh: &AnalysisMeshArtifact,
    node_values: &[[f64; 3]],
) -> Result<Vec<BoundaryFaceVectorValue>, FieldMappingError> {
    validate_nodal_vector_to_boundary_face_topology(mesh)?;
    let node_values_by_id = validate_nodal_vector_field(mesh, node_values)?;

    mesh.boundary_faces
        .iter()
        .map(|face| {
            if face.node_ids.is_empty() {
                return Err(FieldMappingError::BoundaryFaceHasNoNodes {
                    face_id: face.face_id.clone(),
                });
            }
            let mut value_sum = [0.0_f64; 3];
            for node_id in &face.node_ids {
                let Some(value) = node_values_by_id.get(node_id).copied() else {
                    return Err(FieldMappingError::BoundaryFaceReferencesUnknownNode {
                        face_id: face.face_id.clone(),
                        node_id: *node_id,
                    });
                };
                for component in 0..3 {
                    value_sum[component] += value[component];
                }
            }
            let node_count = face.node_ids.len() as f64;
            Ok(BoundaryFaceVectorValue {
                face_id: face.face_id.clone(),
                value: [
                    value_sum[0] / node_count,
                    value_sum[1] / node_count,
                    value_sum[2] / node_count,
                ],
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
