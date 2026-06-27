use std::collections::BTreeSet;

use crate::{
    artifact::{AnalysisMeshArtifact, ANALYSIS_MESH_SCHEMA_VERSION},
    quality::QualityThresholds,
};

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisMeshValidationOptions {
    pub quality: QualityThresholds,
    pub expected_bounds_m: Option<[[f64; 3]; 2]>,
    pub min_bounds_coverage_ratio: f64,
    pub required_boundary_region_ids: Vec<String>,
    pub required_material_region_ids: Vec<String>,
}

impl Default for AnalysisMeshValidationOptions {
    fn default() -> Self {
        Self {
            quality: QualityThresholds::default(),
            expected_bounds_m: None,
            min_bounds_coverage_ratio: 0.90,
            required_boundary_region_ids: Vec::new(),
            required_material_region_ids: Vec::new(),
        }
    }
}

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
    BoundsCoverageFailed {
        axis: usize,
        coverage_ratio: String,
        required_ratio: String,
    },
    MissingRequiredBoundaryRegion {
        region_id: String,
    },
    MissingRequiredMaterialRegion {
        region_id: String,
    },
}

pub fn validate_analysis_mesh(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    validate_analysis_mesh_with_options(
        mesh,
        AnalysisMeshValidationOptions {
            quality: thresholds,
            ..AnalysisMeshValidationOptions::default()
        },
    )
}

pub fn validate_analysis_mesh_with_options(
    mesh: &AnalysisMeshArtifact,
    options: AnalysisMeshValidationOptions,
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

    validate_required_boundary_regions(mesh, &options.required_boundary_region_ids)?;
    validate_required_material_regions(mesh, &options.required_material_region_ids)?;
    validate_bounds_coverage(
        mesh,
        options.expected_bounds_m,
        options.min_bounds_coverage_ratio,
    )?;
    validate_quality(mesh, options.quality)
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

fn validate_required_boundary_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .boundary_faces
        .iter()
        .flat_map(|face| face.region_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
                region_id: region_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_required_material_regions(
    mesh: &AnalysisMeshArtifact,
    required_region_ids: &[String],
) -> Result<(), AnalysisMeshValidationError> {
    if required_region_ids.is_empty() {
        return Ok(());
    }
    let present = mesh
        .volume_elements
        .iter()
        .map(|element| element.material_region_id.as_str())
        .collect::<BTreeSet<_>>();
    for region_id in required_region_ids {
        if !present.contains(region_id.as_str()) {
            return Err(AnalysisMeshValidationError::MissingRequiredMaterialRegion {
                region_id: region_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_bounds_coverage(
    mesh: &AnalysisMeshArtifact,
    expected_bounds_m: Option<[[f64; 3]; 2]>,
    min_bounds_coverage_ratio: f64,
) -> Result<(), AnalysisMeshValidationError> {
    let Some(expected) = expected_bounds_m else {
        return Ok(());
    };
    if !min_bounds_coverage_ratio.is_finite() || min_bounds_coverage_ratio <= 0.0 {
        return Ok(());
    }
    let Some(actual) = mesh_bounds_m(mesh) else {
        return Ok(());
    };
    for axis in 0..3 {
        let expected_min = expected[0][axis].min(expected[1][axis]);
        let expected_max = expected[0][axis].max(expected[1][axis]);
        if !expected_min.is_finite() || !expected_max.is_finite() {
            continue;
        }
        let expected_span = expected_max - expected_min;
        if expected_span <= f64::EPSILON {
            continue;
        }
        let actual_min = actual[0][axis].min(actual[1][axis]);
        let actual_max = actual[0][axis].max(actual[1][axis]);
        let overlap = (actual_max.min(expected_max) - actual_min.max(expected_min)).max(0.0);
        let coverage = overlap / expected_span;
        if coverage + 1.0e-9 < min_bounds_coverage_ratio {
            return Err(AnalysisMeshValidationError::BoundsCoverageFailed {
                axis,
                coverage_ratio: format!("{coverage:.6}"),
                required_ratio: format!("{min_bounds_coverage_ratio:.6}"),
            });
        }
    }
    Ok(())
}

fn mesh_bounds_m(mesh: &AnalysisMeshArtifact) -> Option<[[f64; 3]; 2]> {
    let mut nodes = mesh.nodes.iter();
    let first = nodes.next()?.coordinates_m;
    let mut min = first;
    let mut max = first;
    for node in nodes {
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    Some([min, max])
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
            adaptive_iterations: Vec::new(),
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

    #[test]
    fn rejects_mesh_that_underfills_expected_bounds() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                expected_bounds_m: Some([[0.0, 0.0, 0.0], [4.0, 1.0, 1.0]]),
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("mesh should fail bounds coverage");
        assert_eq!(
            err,
            AnalysisMeshValidationError::BoundsCoverageFailed {
                axis: 0,
                coverage_ratio: "0.250000".to_string(),
                required_ratio: "0.900000".to_string(),
            }
        );
    }

    #[test]
    fn rejects_missing_required_boundary_region() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_boundary_region_ids: vec!["loaded".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing boundary region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredBoundaryRegion {
                region_id: "loaded".to_string()
            }
        );
    }

    #[test]
    fn rejects_missing_required_material_region() {
        let mesh = valid_tet_mesh();
        let err = validate_analysis_mesh_with_options(
            &mesh,
            AnalysisMeshValidationOptions {
                required_material_region_ids: vec!["rib".to_string()],
                ..AnalysisMeshValidationOptions::default()
            },
        )
        .expect_err("missing material region should fail");
        assert_eq!(
            err,
            AnalysisMeshValidationError::MissingRequiredMaterialRegion {
                region_id: "rib".to_string()
            }
        );
    }
}
