use serde::{Deserialize, Serialize};

use crate::quality::QualityThresholds;

use super::{backend::MeshBackendKind, topology::VolumeElementKind};

pub use runmat_meshing_size::refinement::{
    AdaptiveMeshingOptions, MeshRefinementOptions, RefinementConvergenceOptions,
    RefinementFocusLevel, RefinementFocusOptions, RefinementIndicatorMode,
    RefinementIndicatorOverrides, RefinementStrategy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshKindRequest {
    Solid,
    Shell,
    Beam,
    Surrogate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshElementOrder {
    Linear,
    Quadratic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshProfile {
    Coarse,
    AnalysisReady,
    Adaptive,
    Fine,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshTargetSize {
    Auto,
    LengthM(f64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshValidationPolicyOptions {
    #[serde(default)]
    pub quality: QualityThresholds,
    pub min_bounds_coverage_ratio: f64,
    pub min_volume_coverage_ratio: f64,
    pub min_boundary_area_ratio: f64,
    pub min_boundary_face_recovery_ratio: f64,
    pub min_boundary_edge_recovery_ratio: f64,
    #[serde(default)]
    pub max_volume_component_count: Option<usize>,
}

impl Default for MeshValidationPolicyOptions {
    fn default() -> Self {
        Self {
            quality: QualityThresholds::default(),
            min_bounds_coverage_ratio: 0.90,
            min_volume_coverage_ratio: 0.90,
            min_boundary_area_ratio: 0.90,
            min_boundary_face_recovery_ratio: 1.0,
            min_boundary_edge_recovery_ratio: 1.0,
            max_volume_component_count: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeMeshingOptions {
    pub backend: MeshBackendKind,
    pub kind: MeshKindRequest,
    pub element: VolumeElementKind,
    pub element_order: MeshElementOrder,
    pub profile: MeshProfile,
    pub max_elements: usize,
    pub target_size: MeshTargetSize,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub growth_rate: Option<f64>,
    pub refinement: MeshRefinementOptions,
    #[serde(default)]
    pub validation: MeshValidationPolicyOptions,
}

impl Default for VolumeMeshingOptions {
    fn default() -> Self {
        Self {
            backend: MeshBackendKind::Auto,
            kind: MeshKindRequest::Solid,
            element: VolumeElementKind::Tetrahedron4,
            element_order: MeshElementOrder::Linear,
            profile: MeshProfile::AnalysisReady,
            max_elements: 250_000,
            target_size: MeshTargetSize::Auto,
            min_size_m: None,
            max_size_m: None,
            growth_rate: None,
            refinement: MeshRefinementOptions::default(),
            validation: MeshValidationPolicyOptions::default(),
        }
    }
}
