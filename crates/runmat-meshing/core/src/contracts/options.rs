use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Serialize};

use crate::quality::QualityThresholds;

use super::{backend::MeshBackendKind, topology::VolumeElementKind};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementIndicatorMode {
    Auto,
    On,
    Off,
}

impl<'de> Deserialize<'de> for RefinementIndicatorMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ModeVisitor;

        impl Visitor<'_> for ModeVisitor {
            type Value = RefinementIndicatorMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("auto, on, off, true, or false")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(if value {
                    RefinementIndicatorMode::On
                } else {
                    RefinementIndicatorMode::Off
                })
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => Ok(RefinementIndicatorMode::Auto),
                    "on" | "true" => Ok(RefinementIndicatorMode::On),
                    "off" | "false" => Ok(RefinementIndicatorMode::Off),
                    other => Err(E::custom(format!(
                        "invalid refinement indicator mode `{other}`, expected auto, on, or off"
                    ))),
                }
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }
        }

        deserializer.deserialize_any(ModeVisitor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RefinementIndicatorOverrides {
    #[serde(default, flatten)]
    pub namespaces: BTreeMap<String, BTreeMap<String, RefinementIndicatorMode>>,
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
    #[serde(default)]
    pub indicators: RefinementIndicatorOverrides,
}

impl Default for MeshRefinementOptions {
    fn default() -> Self {
        Self {
            strategy: RefinementStrategy::Auto,
            max_iterations: 4,
            convergence: RefinementConvergenceOptions::default(),
            focus: RefinementFocusOptions::default(),
            indicators: RefinementIndicatorOverrides::default(),
        }
    }
}

pub type AdaptiveMeshingOptions = MeshRefinementOptions;

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
    #[serde(default)]
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
