use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SliverRecoveryOptions {
    pub sliver_aspect_ratio: f64,
    pub min_exact_scaled_jacobian: f64,
    pub exact_quality_tolerance: f64,
}

impl Default for SliverRecoveryOptions {
    fn default() -> Self {
        Self {
            sliver_aspect_ratio: 20.0,
            min_exact_scaled_jacobian: 0.15,
            exact_quality_tolerance: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SliverTetrahedronQuality {
    pub tetrahedron_id: u32,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverClassificationReason {
    AspectRatioOnly,
    AspectRatioAndExactQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliverClassification {
    pub tetrahedron_id: u32,
    pub aspect_ratio: f64,
    pub exact_scaled_jacobian: f64,
    pub reason: SliverClassificationReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverRemovalRejectionReason {
    NoInitialSlivers,
    SliverCountNotReduced,
    ExactQualityViolationRegressed,
    MinimumExactQualityRegressed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SliverRemovalEvaluation {
    pub accepted: bool,
    pub initial_sliver_count: usize,
    pub final_sliver_count: usize,
    pub removed_sliver_count: usize,
    pub initial_exact_quality_violation_count: usize,
    pub final_exact_quality_violation_count: usize,
    pub initial_min_exact_scaled_jacobian: f64,
    pub final_min_exact_scaled_jacobian: f64,
    pub initial_max_aspect_ratio: f64,
    pub final_max_aspect_ratio: f64,
    #[serde(default)]
    pub rejection_reason: Option<SliverRemovalRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliverRecoveryError {
    InvalidOptions,
    NonFiniteQuality { tetrahedron_id: u32 },
}
