use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::artifact::AnalysisMeshArtifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFaceScalarValue {
    pub face_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryNodeVectorValue {
    pub node_id: u32,
    pub value: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFaceVectorValue {
    pub face_id: String,
    pub value: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMappingError {
    ElementFieldLengthMismatch {
        element_value_count: usize,
        volume_element_count: usize,
    },
    NodeVectorFieldLengthMismatch {
        node_value_count: usize,
        node_count: usize,
    },
    NonFiniteElementValue {
        element_index: usize,
    },
    NonFiniteNodeVectorValue {
        node_index: usize,
        component_index: usize,
    },
    BoundaryFaceMissingAdjacentVolume {
        face_id: String,
    },
    BoundaryFaceReferencesUnknownVolume {
        face_id: String,
        volume_element_id: String,
    },
    BoundaryFaceReferencesUnknownNode {
        face_id: String,
        node_id: u32,
    },
    BoundaryFaceHasNoNodes {
        face_id: String,
    },
    BoundaryEdgeReferencesUnknownNode {
        edge_id: String,
        node_id: u32,
    },
}

impl std::fmt::Display for FieldMappingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElementFieldLengthMismatch {
                element_value_count,
                volume_element_count,
            } => write!(
                formatter,
                "element scalar field length {element_value_count} does not match volume element count {volume_element_count}"
            ),
            Self::NodeVectorFieldLengthMismatch {
                node_value_count,
                node_count,
            } => write!(
                formatter,
                "node vector field length {node_value_count} does not match mesh node count {node_count}"
            ),
            Self::NonFiniteElementValue { element_index } => {
                write!(formatter, "element scalar field value {element_index} is not finite")
            }
            Self::NonFiniteNodeVectorValue {
                node_index,
                component_index,
            } => write!(
                formatter,
                "node vector field value {node_index} component {component_index} is not finite"
            ),
            Self::BoundaryFaceMissingAdjacentVolume { face_id } => {
                write!(formatter, "boundary face {face_id} has no adjacent volume element")
            }
            Self::BoundaryFaceReferencesUnknownVolume {
                face_id,
                volume_element_id,
            } => write!(
                formatter,
                "boundary face {face_id} references unknown volume element {volume_element_id}"
            ),
            Self::BoundaryFaceReferencesUnknownNode { face_id, node_id } => write!(
                formatter,
                "boundary face {face_id} references unknown node {node_id}"
            ),
            Self::BoundaryFaceHasNoNodes { face_id } => {
                write!(formatter, "boundary face {face_id} has no nodes")
            }
            Self::BoundaryEdgeReferencesUnknownNode { edge_id, node_id } => write!(
                formatter,
                "boundary edge {edge_id} references unknown node {node_id}"
            ),
        }
    }
}

impl std::error::Error for FieldMappingError {}

pub fn map_volume_scalar_field_to_boundary_faces(
    mesh: &AnalysisMeshArtifact,
    element_values: &[f64],
) -> Result<Vec<BoundaryFaceScalarValue>, FieldMappingError> {
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

fn validate_nodal_vector_field<'a>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
            MeshBackendSummary,
        },
        provenance::AnalysisMeshProvenance,
        quality::AnalysisMeshQualityReport,
        size::field::MeshSizingField,
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    #[test]
    fn maps_element_scalar_values_to_boundary_faces() {
        let mesh = field_mapping_mesh();

        let values = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
            .expect("boundary scalar mapping should succeed");

        assert_eq!(
            values,
            vec![
                BoundaryFaceScalarValue {
                    face_id: "bf1".to_string(),
                    value: 10.0,
                },
                BoundaryFaceScalarValue {
                    face_id: "bf2".to_string(),
                    value: 15.0,
                },
            ]
        );
    }

    #[test]
    fn maps_nodal_vector_values_to_boundary_nodes() {
        let mesh = field_mapping_mesh();

        let values = map_nodal_vector_field_to_boundary_nodes(&mesh, &nodal_vector_values())
            .expect("boundary node vector mapping should succeed");

        assert_eq!(
            values,
            vec![
                BoundaryNodeVectorValue {
                    node_id: 1,
                    value: [1.0, 0.0, 0.0],
                },
                BoundaryNodeVectorValue {
                    node_id: 2,
                    value: [2.0, 0.0, 0.0],
                },
                BoundaryNodeVectorValue {
                    node_id: 3,
                    value: [3.0, 0.0, 0.0],
                },
                BoundaryNodeVectorValue {
                    node_id: 4,
                    value: [4.0, 0.0, 0.0],
                },
            ]
        );
    }

    #[test]
    fn maps_nodal_vector_values_to_boundary_faces() {
        let mesh = field_mapping_mesh();

        let values = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
            .expect("boundary face vector mapping should succeed");

        assert_eq!(
            values,
            vec![
                BoundaryFaceVectorValue {
                    face_id: "bf1".to_string(),
                    value: [7.0 / 3.0, 0.0, 0.0],
                },
                BoundaryFaceVectorValue {
                    face_id: "bf2".to_string(),
                    value: [2.0, 0.0, 0.0],
                },
            ]
        );
    }

    #[test]
    fn rejects_unmapped_boundary_faces() {
        let mut mesh = field_mapping_mesh();
        mesh.boundary_faces[0].adjacent_volume_element_ids.clear();

        let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
            .expect_err("missing adjacency should fail");

        assert_eq!(
            err,
            FieldMappingError::BoundaryFaceMissingAdjacentVolume {
                face_id: "bf1".to_string(),
            }
        );
    }

    #[test]
    fn rejects_element_field_length_mismatch() {
        let mesh = field_mapping_mesh();

        let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0])
            .expect_err("field length mismatch should fail");

        assert_eq!(
            err,
            FieldMappingError::ElementFieldLengthMismatch {
                element_value_count: 1,
                volume_element_count: 2,
            }
        );
    }

    #[test]
    fn rejects_node_vector_field_length_mismatch() {
        let mesh = field_mapping_mesh();

        let err = map_nodal_vector_field_to_boundary_nodes(&mesh, &[[1.0, 0.0, 0.0]])
            .expect_err("node field length mismatch should fail");

        assert_eq!(
            err,
            FieldMappingError::NodeVectorFieldLengthMismatch {
                node_value_count: 1,
                node_count: 5,
            }
        );
    }

    #[test]
    fn rejects_nonfinite_node_vector_values() {
        let mesh = field_mapping_mesh();
        let mut values = nodal_vector_values();
        values[2][1] = f64::INFINITY;

        let err = map_nodal_vector_field_to_boundary_faces(&mesh, &values)
            .expect_err("nonfinite node vector should fail");

        assert_eq!(
            err,
            FieldMappingError::NonFiniteNodeVectorValue {
                node_index: 2,
                component_index: 1,
            }
        );
    }

    #[test]
    fn rejects_nonfinite_element_values() {
        let mesh = field_mapping_mesh();

        let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, f64::NAN])
            .expect_err("nonfinite element value should fail");

        assert_eq!(
            err,
            FieldMappingError::NonFiniteElementValue { element_index: 1 }
        );
    }

    #[test]
    fn rejects_boundary_faces_referencing_unknown_volume_elements() {
        let mut mesh = field_mapping_mesh();
        mesh.boundary_faces[0].adjacent_volume_element_ids = vec!["missing".to_string()];

        let err = map_volume_scalar_field_to_boundary_faces(&mesh, &[10.0, 20.0])
            .expect_err("unknown adjacent volume element should fail");

        assert_eq!(
            err,
            FieldMappingError::BoundaryFaceReferencesUnknownVolume {
                face_id: "bf1".to_string(),
                volume_element_id: "missing".to_string(),
            }
        );
    }

    #[test]
    fn rejects_boundary_faces_referencing_unknown_nodes() {
        let mut mesh = field_mapping_mesh();
        mesh.boundary_faces[0].node_ids = vec![1, 2, 99];

        let err = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
            .expect_err("unknown boundary face node should fail");

        assert_eq!(
            err,
            FieldMappingError::BoundaryFaceReferencesUnknownNode {
                face_id: "bf1".to_string(),
                node_id: 99,
            }
        );
    }

    #[test]
    fn rejects_boundary_edges_referencing_unknown_nodes() {
        let mut mesh = field_mapping_mesh();
        mesh.boundary_edges
            .push(crate::artifact::AnalysisBoundaryEdge {
                edge_id: "be1".to_string(),
                node_ids: [1, 99],
                adjacent_boundary_face_ids: Vec::new(),
                region_ids: Vec::new(),
                provenance: Vec::new(),
            });

        let err = map_nodal_vector_field_to_boundary_nodes(&mesh, &nodal_vector_values())
            .expect_err("unknown boundary edge node should fail");

        assert_eq!(
            err,
            FieldMappingError::BoundaryEdgeReferencesUnknownNode {
                edge_id: "be1".to_string(),
                node_id: 99,
            }
        );
    }

    #[test]
    fn rejects_boundary_faces_without_nodes() {
        let mut mesh = field_mapping_mesh();
        mesh.boundary_faces[0].node_ids.clear();

        let err = map_nodal_vector_field_to_boundary_faces(&mesh, &nodal_vector_values())
            .expect_err("empty boundary face should fail");

        assert_eq!(
            err,
            FieldMappingError::BoundaryFaceHasNoNodes {
                face_id: "bf1".to_string(),
            }
        );
    }

    fn nodal_vector_values() -> Vec<[f64; 3]> {
        vec![
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    }

    fn field_mapping_mesh() -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: "analysis-mesh/v1".to_string(),
            mesh_id: "field_mapping_fixture".to_string(),
            nodes: vec![
                AnalysisMeshNode {
                    node_id: 1,
                    coordinates_m: [0.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 2,
                    coordinates_m: [1.0, 0.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 3,
                    coordinates_m: [0.0, 1.0, 0.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 4,
                    coordinates_m: [0.0, 0.0, 1.0],
                    provenance: Vec::new(),
                },
                AnalysisMeshNode {
                    node_id: 5,
                    coordinates_m: [0.0, 0.0, -1.0],
                    provenance: Vec::new(),
                },
            ],
            volume_elements: vec![
                AnalysisVolumeElement {
                    element_id: "e1".to_string(),
                    kind: VolumeElementKind::Tet4,
                    node_ids: vec![1, 2, 3, 4],
                    material_region_id: "mat".to_string(),
                    provenance: Vec::new(),
                },
                AnalysisVolumeElement {
                    element_id: "e2".to_string(),
                    kind: VolumeElementKind::Tet4,
                    node_ids: vec![1, 3, 2, 5],
                    material_region_id: "mat".to_string(),
                    provenance: Vec::new(),
                },
            ],
            boundary_faces: vec![
                AnalysisBoundaryFace {
                    face_id: "bf1".to_string(),
                    kind: BoundaryElementKind::Tri3,
                    node_ids: vec![1, 2, 4],
                    adjacent_volume_element_ids: vec!["e1".to_string()],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
                AnalysisBoundaryFace {
                    face_id: "bf2".to_string(),
                    kind: BoundaryElementKind::Tri3,
                    node_ids: vec![1, 2, 3],
                    adjacent_volume_element_ids: vec!["e1".to_string(), "e2".to_string()],
                    region_ids: Vec::new(),
                    provenance: Vec::new(),
                },
            ],
            boundary_edges: Vec::new(),
            quality: AnalysisMeshQualityReport::default(),
            sizing: MeshSizingField::default(),
            backend: MeshBackendSummary::default(),
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "fixture".to_string(),
                source_geometry_id: "field_mapping_fixture".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }
}
