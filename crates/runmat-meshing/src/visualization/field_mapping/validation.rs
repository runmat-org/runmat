use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{
    AnalysisFieldTopologyLocation, AnalysisMeshArtifact, ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID,
    ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID, ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
    TETRAHEDRON4_FIELD_ELEMENT_KIND, TRI3_FIELD_ELEMENT_KIND,
};

use super::FieldMappingError;

pub(super) fn validate_volume_scalar_to_boundary_face_topology(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), FieldMappingError> {
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::VolumeElement,
        Some(TETRAHEDRON4_FIELD_ELEMENT_KIND),
        mesh.volume_elements.len(),
    )?;
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::BoundaryFace,
        Some(TRI3_FIELD_ELEMENT_KIND),
        mesh.boundary_faces.len(),
    )
}

pub(super) fn validate_nodal_vector_to_boundary_node_topology(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), FieldMappingError> {
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::Node,
        None,
        mesh.nodes.len(),
    )?;
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::BoundaryFace,
        Some(TRI3_FIELD_ELEMENT_KIND),
        mesh.boundary_faces.len(),
    )?;
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::BoundaryEdge,
        None,
        mesh.boundary_edges.len(),
    )
}

pub(super) fn validate_nodal_vector_to_boundary_face_topology(
    mesh: &AnalysisMeshArtifact,
) -> Result<(), FieldMappingError> {
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_FIELD_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::Node,
        None,
        mesh.nodes.len(),
    )?;
    validate_field_topology_count(
        mesh,
        ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID,
        AnalysisFieldTopologyLocation::BoundaryFace,
        Some(TRI3_FIELD_ELEMENT_KIND),
        mesh.boundary_faces.len(),
    )
}

fn validate_field_topology_count(
    mesh: &AnalysisMeshArtifact,
    topology_id: &str,
    location: AnalysisFieldTopologyLocation,
    element_kind: Option<&str>,
    actual_entity_count: usize,
) -> Result<(), FieldMappingError> {
    let same_location = mesh
        .field_topology
        .iter()
        .filter(|descriptor| {
            descriptor.topology_id == topology_id && descriptor.location == location
        })
        .collect::<Vec<_>>();
    if same_location.is_empty() {
        return Err(FieldMappingError::FieldTopologyMissing {
            topology_id: topology_id.to_string(),
            location,
            element_kind: element_kind.map(str::to_string),
        });
    }

    let descriptor = same_location
        .iter()
        .copied()
        .find(|descriptor| descriptor.element_kind.as_deref() == element_kind)
        .ok_or_else(|| FieldMappingError::FieldTopologyElementKindMismatch {
            topology_id: topology_id.to_string(),
            location,
            expected_element_kind: element_kind.unwrap_or("none").to_string(),
            actual_element_kind: same_location
                .first()
                .and_then(|descriptor| descriptor.element_kind.clone()),
        })?;

    if descriptor.entity_count != actual_entity_count {
        return Err(FieldMappingError::FieldTopologyCountMismatch {
            topology_id: topology_id.to_string(),
            location,
            expected_entity_count: descriptor.entity_count,
            actual_entity_count,
        });
    }

    Ok(())
}

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
