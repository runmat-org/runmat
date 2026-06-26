use std::collections::BTreeSet;

use crate::{
    artifact::{AnalysisMeshArtifact, ANALYSIS_MESH_SCHEMA_VERSION},
    quality::QualityThresholds,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisMeshValidationError {
    UnsupportedSchema {
        schema_version: String,
    },
    EmptyNodes,
    EmptyVolumeElements,
    DuplicateNodeId {
        node_id: u32,
    },
    NonFiniteNodeCoordinate {
        node_id: u32,
    },
    DuplicateElementId {
        element_id: String,
    },
    UnsupportedVolumeElementKind {
        element_id: String,
    },
    WrongVolumeElementNodeCount {
        element_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownVolumeElementNode {
        element_id: String,
        node_id: u32,
    },
    RepeatedVolumeElementNode {
        element_id: String,
    },
    MissingMaterialRegion {
        element_id: String,
    },
    DuplicateBoundaryFaceId {
        face_id: String,
    },
    UnsupportedBoundaryElementKind {
        face_id: String,
    },
    WrongBoundaryFaceNodeCount {
        face_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownBoundaryFaceNode {
        face_id: String,
        node_id: u32,
    },
    RepeatedBoundaryFaceNode {
        face_id: String,
    },
    UnknownBoundaryAdjacentElement {
        face_id: String,
        element_id: String,
    },
    QualityThresholdFailed {
        reason: String,
    },
}

pub fn validate_analysis_mesh(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.schema_version != ANALYSIS_MESH_SCHEMA_VERSION {
        return Err(AnalysisMeshValidationError::UnsupportedSchema {
            schema_version: mesh.schema_version.clone(),
        });
    }
    if mesh.nodes.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyNodes);
    }
    if mesh.volume_elements.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyVolumeElements);
    }

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

    let mut element_ids = BTreeSet::<String>::new();
    for element in &mesh.volume_elements {
        if !element_ids.insert(element.element_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateElementId {
                element_id: element.element_id.clone(),
            });
        }
        if !element.kind.is_supported_for_solid_solve() {
            return Err(AnalysisMeshValidationError::UnsupportedVolumeElementKind {
                element_id: element.element_id.clone(),
            });
        }
        let expected = element.kind.node_count();
        if element.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongVolumeElementNodeCount {
                element_id: element.element_id.clone(),
                expected,
                actual: element.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &element.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownVolumeElementNode {
                    element_id: element.element_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedVolumeElementNode {
                    element_id: element.element_id.clone(),
                });
            }
        }
        if element.material_region_id.trim().is_empty() {
            return Err(AnalysisMeshValidationError::MissingMaterialRegion {
                element_id: element.element_id.clone(),
            });
        }
    }

    let mut face_ids = BTreeSet::<String>::new();
    for face in &mesh.boundary_faces {
        if !face_ids.insert(face.face_id.clone()) {
            return Err(AnalysisMeshValidationError::DuplicateBoundaryFaceId {
                face_id: face.face_id.clone(),
            });
        }
        if !face.kind.is_supported_for_boundary_mapping() {
            return Err(
                AnalysisMeshValidationError::UnsupportedBoundaryElementKind {
                    face_id: face.face_id.clone(),
                },
            );
        }
        let expected = face.kind.node_count();
        if face.node_ids.len() != expected {
            return Err(AnalysisMeshValidationError::WrongBoundaryFaceNodeCount {
                face_id: face.face_id.clone(),
                expected,
                actual: face.node_ids.len(),
            });
        }
        let mut local_nodes = BTreeSet::<u32>::new();
        for node_id in &face.node_ids {
            if !node_ids.contains(node_id) {
                return Err(AnalysisMeshValidationError::UnknownBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                    node_id: *node_id,
                });
            }
            if !local_nodes.insert(*node_id) {
                return Err(AnalysisMeshValidationError::RepeatedBoundaryFaceNode {
                    face_id: face.face_id.clone(),
                });
            }
        }
        for element_id in &face.adjacent_volume_element_ids {
            if !element_ids.contains(element_id) {
                return Err(
                    AnalysisMeshValidationError::UnknownBoundaryAdjacentElement {
                        face_id: face.face_id.clone(),
                        element_id: element_id.clone(),
                    },
                );
            }
        }
    }

    validate_quality(mesh, thresholds)
}

fn validate_quality(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    if !mesh.quality.min_scaled_jacobian.is_finite()
        || mesh.quality.min_scaled_jacobian < thresholds.min_scaled_jacobian
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "min_scaled_jacobian".to_string(),
        });
    }
    if !mesh.quality.max_aspect_ratio.is_finite()
        || mesh.quality.max_aspect_ratio > thresholds.max_aspect_ratio
    {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "max_aspect_ratio".to_string(),
        });
    }
    if !thresholds.allow_inverted_elements && mesh.quality.inverted_element_count > 0 {
        return Err(AnalysisMeshValidationError::QualityThresholdFailed {
            reason: "inverted_element_count".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
            ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::AnalysisMeshQualityReport,
        sizing::MeshSizingField,
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    fn valid_tet_mesh() -> AnalysisMeshArtifact {
        AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_valid".to_string(),
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
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "e1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "mat_region".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "f1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["e1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: Vec::new(),
            quality: AnalysisMeshQualityReport::default(),
            sizing: MeshSizingField::default(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        }
    }

    #[test]
    fn accepts_minimal_valid_tet4_mesh() {
        let mesh = valid_tet_mesh();
        validate_analysis_mesh(&mesh, QualityThresholds::default()).expect("mesh should validate");
    }

    #[test]
    fn rejects_empty_volume_elements() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements.clear();
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("empty volume elements should fail");
        assert_eq!(err, AnalysisMeshValidationError::EmptyVolumeElements);
    }

    #[test]
    fn rejects_unsupported_element_kind_until_assembly_exists() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements[0].kind = VolumeElementKind::Hex8;
        mesh.volume_elements[0].node_ids = vec![1, 2, 3, 4, 1, 2, 3, 4];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unsupported element kind should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnsupportedVolumeElementKind {
                element_id: "e1".to_string()
            }
        );
    }

    #[test]
    fn rejects_missing_material_coverage() {
        let mut mesh = valid_tet_mesh();
        mesh.volume_elements[0].material_region_id.clear();
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("missing material region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingMaterialRegion {
                element_id: "e1".to_string()
            }
        );
    }

    #[test]
    fn rejects_unmapped_boundary_nodes() {
        let mut mesh = valid_tet_mesh();
        mesh.boundary_faces[0].node_ids = vec![1, 2, 99];
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("unknown boundary node should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::UnknownBoundaryFaceNode {
                face_id: "f1".to_string(),
                node_id: 99
            }
        );
    }

    #[test]
    fn rejects_quality_threshold_failures() {
        let mut mesh = valid_tet_mesh();
        mesh.quality.min_scaled_jacobian = 0.01;
        let err = validate_analysis_mesh(&mesh, QualityThresholds::default())
            .expect_err("low jacobian should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::QualityThresholdFailed {
                reason: "min_scaled_jacobian".to_string()
            }
        );
    }
}
