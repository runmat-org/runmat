use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    artifact::{AnalysisMeshArtifact, AnalysisVolumeElement},
    topology::VolumeElementKind,
};

pub const MODULE_PURPOSE: &str = "compact solid evidence summaries";

pub use crate::contracts::StageEvidence;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshQualityEvidence {
    pub min_scaled_jacobian: f64,
    #[serde(default)]
    pub min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub scaled_jacobian_p05: Option<f64>,
    #[serde(default)]
    pub scaled_jacobian_p50: Option<f64>,
    #[serde(default)]
    pub scaled_jacobian_p95: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p05: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p50: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p95: Option<f64>,
    pub mean_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    #[serde(default)]
    pub aspect_ratio_p50: Option<f64>,
    #[serde(default)]
    pub aspect_ratio_p95: Option<f64>,
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
    #[serde(default)]
    pub material_region_volume_m3: BTreeMap<String, f64>,
    pub boundary_region_face_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub boundary_region_recovered_face_counts: BTreeMap<String, usize>,
    pub boundary_region_edge_counts: BTreeMap<String, usize>,
}

pub(super) fn quality_evidence(mesh: &AnalysisMeshArtifact) -> MeshQualityEvidence {
    let mut scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut aspect_ratio_bins = BTreeMap::<String, usize>::new();
    let mut volume_bins = BTreeMap::<String, usize>::new();
    let mut scaled_jacobians = Vec::<f64>::new();
    let mut exact_scaled_jacobians = Vec::<f64>::new();
    let mut aspect_ratios = Vec::<f64>::new();
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
        if element.scaled_jacobian.is_finite() {
            scaled_jacobians.push(element.scaled_jacobian);
        }
        if element.exact_scaled_jacobian.is_finite() {
            exact_scaled_jacobians.push(element.exact_scaled_jacobian);
        }
        if element.aspect_ratio.is_finite() {
            aspect_ratios.push(element.aspect_ratio);
        }
    }
    scaled_jacobians.sort_by(f64::total_cmp);
    exact_scaled_jacobians.sort_by(f64::total_cmp);
    aspect_ratios.sort_by(f64::total_cmp);

    MeshQualityEvidence {
        min_scaled_jacobian: mesh.quality.min_scaled_jacobian,
        min_exact_scaled_jacobian: mesh.quality.min_exact_scaled_jacobian,
        scaled_jacobian_p05: percentile(&scaled_jacobians, 0.05),
        scaled_jacobian_p50: percentile(&scaled_jacobians, 0.50),
        scaled_jacobian_p95: percentile(&scaled_jacobians, 0.95),
        exact_scaled_jacobian_p05: percentile(&exact_scaled_jacobians, 0.05),
        exact_scaled_jacobian_p50: percentile(&exact_scaled_jacobians, 0.50),
        exact_scaled_jacobian_p95: percentile(&exact_scaled_jacobians, 0.95),
        mean_aspect_ratio: mesh.quality.mean_aspect_ratio,
        max_aspect_ratio: mesh.quality.max_aspect_ratio,
        aspect_ratio_p50: percentile(&aspect_ratios, 0.50),
        aspect_ratio_p95: percentile(&aspect_ratios, 0.95),
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

pub(super) fn region_evidence(mesh: &AnalysisMeshArtifact) -> MeshRegionEvidence {
    let mut material_region_element_counts = BTreeMap::<String, usize>::new();
    let mut material_region_volume_m3 = BTreeMap::<String, f64>::new();
    for element in &mesh.volume_elements {
        *material_region_element_counts
            .entry(element.material_region_id.clone())
            .or_default() += 1;
        let volume_m3 = element_volume_m3(mesh, element);
        if volume_m3.is_finite() && volume_m3 > 0.0 {
            *material_region_volume_m3
                .entry(element.material_region_id.clone())
                .or_default() += volume_m3;
        }
    }

    let mut boundary_region_face_counts = BTreeMap::<String, usize>::new();
    let mut boundary_region_recovered_face_counts = BTreeMap::<String, usize>::new();
    for face in &mesh.boundary_faces {
        for region_id in &face.region_ids {
            *boundary_region_face_counts
                .entry(region_id.clone())
                .or_default() += 1;
            if !face.adjacent_volume_element_ids.is_empty() {
                *boundary_region_recovered_face_counts
                    .entry(region_id.clone())
                    .or_default() += 1;
            }
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
        material_region_volume_m3,
        boundary_region_face_counts,
        boundary_region_recovered_face_counts,
        boundary_region_edge_counts,
    }
}

fn element_volume_m3(mesh: &AnalysisMeshArtifact, element: &AnalysisVolumeElement) -> f64 {
    if element.kind != VolumeElementKind::Tetrahedron4 || element.node_ids.len() != 4 {
        return 0.0;
    }
    let Some(points) = element_tetrahedron_points(mesh, element.node_ids.as_slice()) else {
        return 0.0;
    };
    tetrahedron_volume_m3(points)
}

fn element_tetrahedron_points(
    mesh: &AnalysisMeshArtifact,
    node_ids: &[u32],
) -> Option<[[f64; 3]; 4]> {
    Some([
        mesh_node(mesh, node_ids[0])?,
        mesh_node(mesh, node_ids[1])?,
        mesh_node(mesh, node_ids[2])?,
        mesh_node(mesh, node_ids[3])?,
    ])
}

fn mesh_node(mesh: &AnalysisMeshArtifact, node_id: u32) -> Option<[f64; 3]> {
    mesh.nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
}

fn tetrahedron_volume_m3(points: [[f64; 3]; 4]) -> f64 {
    let ab = [
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    ];
    let ac = [
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    ];
    let ad = [
        points[3][0] - points[0][0],
        points[3][1] - points[0][1],
        points[3][2] - points[0][2],
    ];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    ((ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0).abs()
}

fn percentile(sorted_values: &[f64], ratio: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let index = ((sorted_values.len() - 1) as f64 * ratio).round() as usize;
    sorted_values.get(index).copied()
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
