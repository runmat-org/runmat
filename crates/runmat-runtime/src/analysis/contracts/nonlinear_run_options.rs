//! Nonlinear execution controls; physical history and solver authority remain in analysis.
use super::{ContactEnforcementOptions, PrecisionMode, QualityPolicy};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisNonlinearRunOptions {
    pub deterministic_mode: bool,
    pub precision_mode: PrecisionMode,
    pub quality_policy: QualityPolicy,
    #[serde(with = "runmat_types::wire_integer")]
    pub increment_count: usize,
    #[serde(with = "runmat_types::wire_integer")]
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub linear_tolerance: f64,
    #[serde(with = "runmat_types::wire_integer")]
    pub max_load_cutbacks: usize,
    pub minimum_load_increment: f64,
    #[serde(with = "runmat_types::wire_integer")]
    pub max_accepted_increments: usize,
    pub line_search: bool,
    #[serde(with = "runmat_types::wire_integer")]
    pub max_line_search_backtracks: usize,
    pub line_search_reduction: f64,
    #[serde(default)]
    pub contact_enforcement: Option<ContactEnforcementOptions>,
    #[serde(default)]
    pub solver_mesh_artifact_path: Option<String>,
}

impl Default for AnalysisNonlinearRunOptions {
    fn default() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 12,
            max_newton_iters: 24,
            tolerance: 1.0e-6,
            linear_tolerance: 1.0e-10,
            max_load_cutbacks: 8,
            minimum_load_increment: 1.0e-8,
            max_accepted_increments: 4096,
            line_search: true,
            max_line_search_backtracks: 6,
            line_search_reduction: 0.5,
            contact_enforcement: None,
            solver_mesh_artifact_path: None,
        }
    }
}

impl AnalysisNonlinearRunOptions {
    pub fn coarse() -> Self {
        Self {
            deterministic_mode: false,
            precision_mode: PrecisionMode::Fp32,
            quality_policy: QualityPolicy::Exploratory,
            increment_count: 8,
            max_newton_iters: 16,
            tolerance: 5.0e-6,
            linear_tolerance: 1.0e-10,
            max_load_cutbacks: 8,
            minimum_load_increment: 1.0e-8,
            max_accepted_increments: 4096,
            line_search: false,
            max_line_search_backtracks: 0,
            line_search_reduction: 0.6,
            contact_enforcement: None,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn high_accuracy() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Strict,
            increment_count: 24,
            max_newton_iters: 40,
            tolerance: 1.0e-7,
            linear_tolerance: 1.0e-10,
            max_load_cutbacks: 8,
            minimum_load_increment: 1.0e-8,
            max_accepted_increments: 4096,
            line_search: true,
            max_line_search_backtracks: 10,
            line_search_reduction: 0.5,
            contact_enforcement: None,
            solver_mesh_artifact_path: None,
        }
    }

    pub fn solid_recommended() -> Self {
        Self {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            increment_count: 24,
            max_newton_iters: 28,
            tolerance: 1.0e-6,
            linear_tolerance: 1.0e-10,
            max_load_cutbacks: 8,
            minimum_load_increment: 1.0e-8,
            max_accepted_increments: 4096,
            line_search: true,
            max_line_search_backtracks: 8,
            line_search_reduction: 0.5,
            contact_enforcement: None,
            solver_mesh_artifact_path: None,
        }
    }
}
