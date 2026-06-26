use serde::{Deserialize, Serialize};

use crate::topology::VolumeElementKind;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementStrategy {
    None,
    Uniform,
    Adaptive,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementFocusLevel {
    Off,
    Normal,
    Fine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementFocusOptions {
    pub loads: RefinementFocusLevel,
    pub constraints: RefinementFocusLevel,
    pub interfaces: RefinementFocusLevel,
    pub curvature: bool,
    pub small_features: bool,
}

impl Default for RefinementFocusOptions {
    fn default() -> Self {
        Self {
            loads: RefinementFocusLevel::Fine,
            constraints: RefinementFocusLevel::Fine,
            interfaces: RefinementFocusLevel::Normal,
            curvature: true,
            small_features: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementConvergenceOptions {
    pub field_change_tolerance: f64,
    pub energy_change_tolerance: f64,
    #[serde(default)]
    pub residual_tolerance: Option<f64>,
}

impl Default for RefinementConvergenceOptions {
    fn default() -> Self {
        Self {
            field_change_tolerance: 0.05,
            energy_change_tolerance: 0.02,
            residual_tolerance: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshRefinementOptions {
    pub strategy: RefinementStrategy,
    pub max_iterations: usize,
    pub convergence: RefinementConvergenceOptions,
    pub focus: RefinementFocusOptions,
}

impl Default for MeshRefinementOptions {
    fn default() -> Self {
        Self {
            strategy: RefinementStrategy::Auto,
            max_iterations: 4,
            convergence: RefinementConvergenceOptions::default(),
            focus: RefinementFocusOptions::default(),
        }
    }
}

pub type AdaptiveMeshingOptions = MeshRefinementOptions;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VolumeMeshingOptions {
    pub kind: MeshKindRequest,
    pub element: VolumeElementKind,
    pub element_order: MeshElementOrder,
    pub profile: MeshProfile,
    pub max_elements: usize,
    pub target_size: MeshTargetSize,
    pub refinement: MeshRefinementOptions,
}

impl Default for VolumeMeshingOptions {
    fn default() -> Self {
        Self {
            kind: MeshKindRequest::Solid,
            element: VolumeElementKind::Tet4,
            element_order: MeshElementOrder::Linear,
            profile: MeshProfile::AnalysisReady,
            max_elements: 250_000,
            target_size: MeshTargetSize::Auto,
            refinement: MeshRefinementOptions::default(),
        }
    }
}
