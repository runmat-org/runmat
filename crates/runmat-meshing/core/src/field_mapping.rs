use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::artifact::AnalysisMeshArtifact;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryFaceScalarValue {
    pub face_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMappingError {
    ElementFieldLengthMismatch {
        element_value_count: usize,
        volume_element_count: usize,
    },
    NonFiniteElementValue {
        element_index: usize,
    },
    BoundaryFaceMissingAdjacentVolume {
        face_id: String,
    },
    BoundaryFaceReferencesUnknownVolume {
        face_id: String,
        volume_element_id: String,
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
            Self::NonFiniteElementValue { element_index } => {
                write!(formatter, "element scalar field value {element_index} is not finite")
            }
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
        sizing::MeshSizingField,
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
