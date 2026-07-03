use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::artifact::AnalysisMeshArtifact;

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
