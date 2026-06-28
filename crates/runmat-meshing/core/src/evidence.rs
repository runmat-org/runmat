use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{AnalysisMeshArtifact, MeshBackendSummary},
    quality::QualityThresholds,
    validation::AnalysisMeshValidationOptions,
};

pub const MESH_EVIDENCE_SCHEMA_VERSION: &str = "mesh-evidence/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshEvidenceArtifact {
    pub schema_version: String,
    pub mesh_id: String,
    pub backend: MeshBackendSummary,
    pub topology: MeshTopologyEvidence,
    pub sizing: MeshSizingEvidence,
    pub quality: MeshQualityEvidence,
    pub regions: MeshRegionEvidence,
    pub validation: MeshValidationEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshTopologyEvidence {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub adaptive_iteration_count: usize,
    pub bounds_min_m: Option<[f64; 3]>,
    pub bounds_max_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSizingEvidence {
    pub global_target_size_m: Option<f64>,
    pub min_size_m: Option<f64>,
    pub max_size_m: Option<f64>,
    pub sample_count: usize,
    pub applied_sample_count: usize,
    pub rejected_sample_count: usize,
    pub inserted_breakpoint_count: usize,
    pub applied_by_reason: BTreeMap<String, usize>,
    pub rejected_by_status: BTreeMap<String, usize>,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshQualityEvidence {
    pub min_scaled_jacobian: f64,
    #[serde(default)]
    pub min_exact_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    pub inverted_element_count: usize,
    pub mean_boundary_projection_error_m: f64,
    pub max_boundary_projection_error_m: f64,
    pub element_quality_sample_count: usize,
    pub scaled_jacobian_bins: BTreeMap<String, usize>,
    #[serde(default)]
    pub exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub aspect_ratio_bins: BTreeMap<String, usize>,
    pub volume_bins: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshRegionEvidence {
    pub material_region_element_counts: BTreeMap<String, usize>,
    pub boundary_region_face_counts: BTreeMap<String, usize>,
    pub boundary_region_edge_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshValidationEvidence {
    pub quality: QualityThresholds,
    pub expected_bounds_m: Option<[[f64; 3]; 2]>,
    pub min_bounds_coverage_ratio: f64,
    pub expected_volume_m3: Option<f64>,
    pub min_volume_coverage_ratio: f64,
    pub expected_boundary_area_m2: Option<f64>,
    pub min_boundary_area_ratio: f64,
    pub min_boundary_face_recovery_ratio: f64,
    pub min_boundary_edge_recovery_ratio: f64,
    pub required_boundary_region_ids: Vec<String>,
    pub required_material_region_ids: Vec<String>,
    pub boundary_recovery: MeshBoundaryRecoveryEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBoundaryRecoveryEvidence {
    pub boundary_face_recovery_ratio: f64,
    pub boundary_edge_recovery_ratio: f64,
    pub recovered_boundary_face_count: usize,
    pub recovered_boundary_edge_count: usize,
}

pub fn build_mesh_evidence_artifact(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshEvidenceArtifact {
    build_mesh_evidence_artifact_with_validation_evidence(
        mesh,
        validation_evidence(mesh, validation),
    )
}

pub fn build_mesh_evidence_artifact_with_validation_evidence(
    mesh: &AnalysisMeshArtifact,
    mut validation: MeshValidationEvidence,
) -> MeshEvidenceArtifact {
    validation.boundary_recovery = boundary_recovery_evidence(mesh);
    MeshEvidenceArtifact {
        schema_version: MESH_EVIDENCE_SCHEMA_VERSION.to_string(),
        mesh_id: mesh.mesh_id.clone(),
        backend: mesh.backend.clone(),
        topology: topology_evidence(mesh),
        sizing: sizing_evidence(mesh),
        quality: quality_evidence(mesh),
        regions: region_evidence(mesh),
        validation,
    }
}

fn topology_evidence(mesh: &AnalysisMeshArtifact) -> MeshTopologyEvidence {
    let bounds = mesh_bounds_m(mesh);
    MeshTopologyEvidence {
        node_count: mesh.nodes.len(),
        volume_element_count: mesh.volume_elements.len(),
        boundary_face_count: mesh.boundary_faces.len(),
        boundary_edge_count: mesh.boundary_edges.len(),
        adaptive_iteration_count: mesh.adaptive_iterations.len(),
        bounds_min_m: bounds.map(|bounds| bounds[0]),
        bounds_max_m: bounds.map(|bounds| bounds[1]),
    }
}

fn sizing_evidence(mesh: &AnalysisMeshArtifact) -> MeshSizingEvidence {
    let mut applied_by_reason = BTreeMap::<String, usize>::new();
    let mut inserted_breakpoint_count = 0_usize;
    for application in &mesh.sizing.applied_samples {
        let reason = application
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *applied_by_reason.entry(reason).or_default() += 1;
        inserted_breakpoint_count += application.inserted_breakpoint_count;
    }

    let mut rejected_by_status = BTreeMap::<String, usize>::new();
    let mut rejected_by_reason = BTreeMap::<String, usize>::new();
    for rejection in &mesh.sizing.rejected_samples {
        *rejected_by_status
            .entry(rejection.status.clone())
            .or_default() += 1;
        let reason = rejection
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *rejected_by_reason.entry(reason).or_default() += 1;
    }

    MeshSizingEvidence {
        global_target_size_m: mesh.sizing.global_target_size_m,
        min_size_m: mesh.sizing.min_size_m,
        max_size_m: mesh.sizing.max_size_m,
        sample_count: mesh.sizing.samples.len(),
        applied_sample_count: mesh.sizing.applied_samples.len(),
        rejected_sample_count: mesh.sizing.rejected_samples.len(),
        inserted_breakpoint_count,
        applied_by_reason,
        rejected_by_status,
        rejected_by_reason,
    }
}

fn quality_evidence(mesh: &AnalysisMeshArtifact) -> MeshQualityEvidence {
    let mut scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut aspect_ratio_bins = BTreeMap::<String, usize>::new();
    let mut volume_bins = BTreeMap::<String, usize>::new();
    for element in &mesh.quality.elements {
        *scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.scaled_jacobian))
            .or_default() += 1;
        *exact_scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.exact_scaled_jacobian))
            .or_default() += 1;
        *aspect_ratio_bins
            .entry(aspect_ratio_bin(element.aspect_ratio))
            .or_default() += 1;
        *volume_bins
            .entry(volume_bin(element.volume_m3))
            .or_default() += 1;
    }

    MeshQualityEvidence {
        min_scaled_jacobian: mesh.quality.min_scaled_jacobian,
        min_exact_scaled_jacobian: mesh.quality.min_exact_scaled_jacobian,
        mean_aspect_ratio: mesh.quality.mean_aspect_ratio,
        max_aspect_ratio: mesh.quality.max_aspect_ratio,
        inverted_element_count: mesh.quality.inverted_element_count,
        mean_boundary_projection_error_m: mesh.quality.mean_boundary_projection_error_m,
        max_boundary_projection_error_m: mesh.quality.max_boundary_projection_error_m,
        element_quality_sample_count: mesh.quality.elements.len(),
        scaled_jacobian_bins,
        exact_scaled_jacobian_bins,
        aspect_ratio_bins,
        volume_bins,
    }
}

fn region_evidence(mesh: &AnalysisMeshArtifact) -> MeshRegionEvidence {
    let mut material_region_element_counts = BTreeMap::<String, usize>::new();
    for element in &mesh.volume_elements {
        *material_region_element_counts
            .entry(element.material_region_id.clone())
            .or_default() += 1;
    }

    let mut boundary_region_face_counts = BTreeMap::<String, usize>::new();
    for face in &mesh.boundary_faces {
        for region_id in &face.region_ids {
            *boundary_region_face_counts
                .entry(region_id.clone())
                .or_default() += 1;
        }
    }

    let mut boundary_region_edge_counts = BTreeMap::<String, usize>::new();
    for edge in &mesh.boundary_edges {
        for region_id in &edge.region_ids {
            *boundary_region_edge_counts
                .entry(region_id.clone())
                .or_default() += 1;
        }
    }

    MeshRegionEvidence {
        material_region_element_counts,
        boundary_region_face_counts,
        boundary_region_edge_counts,
    }
}

fn validation_evidence(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshValidationEvidence {
    MeshValidationEvidence {
        quality: validation.quality,
        expected_bounds_m: validation.expected_bounds_m,
        min_bounds_coverage_ratio: validation.min_bounds_coverage_ratio,
        expected_volume_m3: validation.expected_volume_m3,
        min_volume_coverage_ratio: validation.min_volume_coverage_ratio,
        expected_boundary_area_m2: validation.expected_boundary_area_m2,
        min_boundary_area_ratio: validation.min_boundary_area_ratio,
        min_boundary_face_recovery_ratio: validation.min_boundary_face_recovery_ratio,
        min_boundary_edge_recovery_ratio: validation.min_boundary_edge_recovery_ratio,
        required_boundary_region_ids: validation.required_boundary_region_ids.clone(),
        required_material_region_ids: validation.required_material_region_ids.clone(),
        boundary_recovery: boundary_recovery_evidence(mesh),
    }
}

fn boundary_recovery_evidence(mesh: &AnalysisMeshArtifact) -> MeshBoundaryRecoveryEvidence {
    MeshBoundaryRecoveryEvidence {
        boundary_face_recovery_ratio: boundary_face_recovery_ratio(mesh),
        boundary_edge_recovery_ratio: boundary_edge_recovery_ratio(mesh),
        recovered_boundary_face_count: mesh
            .boundary_faces
            .iter()
            .filter(|face| !face.adjacent_volume_element_ids.is_empty())
            .count(),
        recovered_boundary_edge_count: recovered_boundary_edge_count(mesh),
    }
}

fn mesh_bounds_m(mesh: &AnalysisMeshArtifact) -> Option<[[f64; 3]; 2]> {
    let mut iter = mesh.nodes.iter();
    let first = iter.next()?.coordinates_m;
    let mut min = first;
    let mut max = first;
    for node in iter {
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    Some([min, max])
}

fn boundary_face_recovery_ratio(mesh: &AnalysisMeshArtifact) -> f64 {
    if mesh.boundary_faces.is_empty() {
        return 1.0;
    }
    mesh.boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count() as f64
        / mesh.boundary_faces.len() as f64
}

fn boundary_edge_recovery_ratio(mesh: &AnalysisMeshArtifact) -> f64 {
    let expected_edges = boundary_face_edges(mesh);
    if expected_edges.is_empty() {
        return 1.0;
    }
    recovered_boundary_edge_count(mesh) as f64 / expected_edges.len() as f64
}

fn recovered_boundary_edge_count(mesh: &AnalysisMeshArtifact) -> usize {
    let expected_edges = boundary_face_edges(mesh);
    mesh.boundary_edges
        .iter()
        .filter(|edge| expected_edges.contains(&ordered_edge(edge.node_ids[0], edge.node_ids[1])))
        .count()
}

fn boundary_face_edges(mesh: &AnalysisMeshArtifact) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in &mesh.boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        edges.insert(ordered_edge(face.node_ids[0], face.node_ids[1]));
        edges.insert(ordered_edge(face.node_ids[1], face.node_ids[2]));
        edges.insert(ordered_edge(face.node_ids[2], face.node_ids[0]));
    }
    edges
}

fn ordered_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn scaled_jacobian_bin(value: f64) -> String {
    if value < 0.0 {
        "lt_0".to_string()
    } else if value < 0.15 {
        "0_to_0_15".to_string()
    } else if value < 0.35 {
        "0_15_to_0_35".to_string()
    } else if value < 0.65 {
        "0_35_to_0_65".to_string()
    } else {
        "gte_0_65".to_string()
    }
}

fn aspect_ratio_bin(value: f64) -> String {
    if value < 2.0 {
        "lt_2".to_string()
    } else if value < 5.0 {
        "2_to_5".to_string()
    } else if value < 10.0 {
        "5_to_10".to_string()
    } else if value < 20.0 {
        "10_to_20".to_string()
    } else {
        "gte_20".to_string()
    }
}

fn volume_bin(value: f64) -> String {
    if value <= 0.0 {
        "lte_0".to_string()
    } else if value < 1.0e-12 {
        "lt_1e-12".to_string()
    } else if value < 1.0e-9 {
        "1e-12_to_1e-9".to_string()
    } else if value < 1.0e-6 {
        "1e-9_to_1e-6".to_string()
    } else {
        "gte_1e-6".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshNode, AnalysisVolumeElement,
            ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::{AnalysisMeshQualityReport, ElementQuality},
        sizing::{MeshSizingField, SizingSampleApplication, SizingSampleRejection},
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    #[test]
    fn evidence_summarizes_mesh_without_raw_sizing_samples() {
        let mesh = AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_1".to_string(),
            nodes: vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "tet_1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "solid".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "face_1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["tet_1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: vec![
                boundary_edge("edge_1", [1, 2]),
                boundary_edge("edge_2", [2, 3]),
                boundary_edge("edge_3", [1, 3]),
            ],
            quality: AnalysisMeshQualityReport {
                min_scaled_jacobian: 0.5,
                min_exact_scaled_jacobian: 0.45,
                mean_aspect_ratio: 2.0,
                max_aspect_ratio: 2.0,
                inverted_element_count: 0,
                mean_boundary_projection_error_m: 0.0,
                max_boundary_projection_error_m: 0.0,
                elements: vec![ElementQuality {
                    element_id: "tet_1".to_string(),
                    scaled_jacobian: 0.5,
                    exact_scaled_jacobian: 0.45,
                    aspect_ratio: 2.0,
                    volume_m3: 1.0 / 6.0,
                }],
            },
            sizing: MeshSizingField {
                applied_samples: vec![SizingSampleApplication {
                    position_m: [0.0, 0.0, 0.0],
                    target_size_m: 0.25,
                    inserted_breakpoint_count: 2,
                    reason: Some("load_region".to_string()),
                    detail: Some("sample detail should not be copied".to_string()),
                }],
                rejected_samples: vec![SizingSampleRejection {
                    position_m: [0.1, 0.0, 0.0],
                    target_size_m: 0.1,
                    status: "outside_bounds".to_string(),
                    reason: Some("adaptive".to_string()),
                    detail: Some("rejection detail should not be copied".to_string()),
                }],
                ..MeshSizingField::default()
            },
            backend: MeshBackendSummary {
                backend: "production".to_string(),
                ..MeshBackendSummary::default()
            },
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        };

        let evidence =
            build_mesh_evidence_artifact(&mesh, &AnalysisMeshValidationOptions::default());

        assert_eq!(evidence.schema_version, MESH_EVIDENCE_SCHEMA_VERSION);
        assert_eq!(evidence.topology.node_count, 4);
        assert_eq!(evidence.sizing.inserted_breakpoint_count, 2);
        assert_eq!(
            evidence.sizing.applied_by_reason.get("load_region"),
            Some(&1)
        );
        assert_eq!(
            evidence.sizing.rejected_by_status.get("outside_bounds"),
            Some(&1)
        );
        assert_eq!(
            evidence.regions.boundary_region_face_counts.get("fixed"),
            Some(&1)
        );
        assert_eq!(evidence.quality.min_exact_scaled_jacobian, 0.45);
        assert_eq!(
            evidence
                .quality
                .exact_scaled_jacobian_bins
                .get("0_35_to_0_65"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
            1.0
        );

        let encoded = serde_json::to_value(&evidence).expect("serialize evidence");
        assert!(encoded.get("sizing").is_some());
        assert!(
            encoded
                .to_string()
                .contains("sample detail should not be copied")
                == false
        );
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
        AnalysisMeshNode {
            node_id,
            coordinates_m,
            provenance: Vec::new(),
        }
    }

    fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
        AnalysisBoundaryEdge {
            edge_id: edge_id.to_string(),
            node_ids,
            adjacent_boundary_face_ids: vec!["face_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }
    }
}
