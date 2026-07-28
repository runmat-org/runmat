use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateOptions {
    pub min_exact_scaled_jacobian: f64,
    pub exact_quality_tolerance: f64,
    #[serde(default = "default_require_exact_quality_improvement")]
    pub require_exact_quality_improvement: bool,
    #[serde(default)]
    pub allow_min_exact_quality_regression_above_threshold: bool,
}

impl Default for BoundaryQualityCandidateOptions {
    fn default() -> Self {
        Self {
            min_exact_scaled_jacobian: 0.15,
            exact_quality_tolerance: 1.0e-12,
            require_exact_quality_improvement: default_require_exact_quality_improvement(),
            allow_min_exact_quality_regression_above_threshold: false,
        }
    }
}

fn default_require_exact_quality_improvement() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateConstraints {
    pub boundary_recovery_preserved: bool,
    pub target_volume_preserved: bool,
    pub source_provenance_preserved: bool,
}

impl BoundaryQualityCandidateConstraints {
    pub fn preserved() -> Self {
        Self {
            boundary_recovery_preserved: true,
            target_volume_preserved: true,
            source_provenance_preserved: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryQualityCandidateRejectionReason {
    BoundaryRecoveryRegressed,
    TargetVolumeRegressed,
    SourceProvenanceRegressed,
    ExactQualityViolationRegressed,
    MinimumExactQualityRegressed,
    ExactQualityNotImproved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryQualityCandidateEvaluation {
    pub accepted: bool,
    pub initial_exact_quality_violation_count: usize,
    pub final_exact_quality_violation_count: usize,
    pub initial_min_exact_scaled_jacobian: f64,
    pub final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub rejection_reason: Option<BoundaryQualityCandidateRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryQualityCandidateError {
    InvalidOptions,
    NonFiniteQuality { element_id: String },
}
