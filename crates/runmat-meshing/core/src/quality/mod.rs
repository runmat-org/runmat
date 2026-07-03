pub mod boundary;
pub mod predicate;
pub mod spatial_index;
pub mod tolerance;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementQuality {
    pub element_id: String,
    pub scaled_jacobian: f64,
    #[serde(default)]
    pub exact_scaled_jacobian: f64,
    pub aspect_ratio: f64,
    pub volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMeshQualityReport {
    pub min_scaled_jacobian: f64,
    #[serde(default)]
    pub min_exact_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    pub inverted_element_count: usize,
    #[serde(default)]
    pub mean_boundary_projection_error_m: f64,
    #[serde(default)]
    pub max_boundary_projection_error_m: f64,
    #[serde(default)]
    pub elements: Vec<ElementQuality>,
}

impl Default for AnalysisMeshQualityReport {
    fn default() -> Self {
        Self {
            min_scaled_jacobian: 1.0,
            min_exact_scaled_jacobian: 1.0,
            mean_aspect_ratio: 1.0,
            max_aspect_ratio: 1.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_scaled_jacobian: f64,
    pub max_aspect_ratio: f64,
    #[serde(default = "default_max_boundary_projection_error_m")]
    pub max_boundary_projection_error_m: f64,
    pub allow_inverted_elements: bool,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_scaled_jacobian: 0.15,
            max_aspect_ratio: 20.0,
            max_boundary_projection_error_m: default_max_boundary_projection_error_m(),
            allow_inverted_elements: false,
        }
    }
}

fn default_max_boundary_projection_error_m() -> f64 {
    1.0e-6
}
