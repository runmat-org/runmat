use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use crate::FeaContactEnforcement;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonlinearSolveOptions {
    pub increment_count: usize,
    pub max_newton_iters: usize,
    pub tolerance: f64,
    pub linear_tolerance: f64,
    pub max_load_cutbacks: usize,
    pub minimum_load_increment: f64,
    pub max_accepted_increments: usize,
    pub line_search: bool,
    pub max_line_search_backtracks: usize,
    pub line_search_reduction: f64,
    #[serde(default)]
    pub solver_mesh: Option<SolverMeshArtifact>,
    pub contact_enforcement: Option<FeaContactEnforcement>,
}

impl Default for NonlinearSolveOptions {
    fn default() -> Self {
        Self {
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
            solver_mesh: None,
            contact_enforcement: None,
        }
    }
}

impl NonlinearSolveOptions {
    pub(crate) fn newton_policy(&self) -> super::solid::SolidNewtonPolicy {
        super::solid::SolidNewtonPolicy {
            max_iterations: self.max_newton_iters,
            relative_residual_tolerance: self.tolerance,
            linear_relative_tolerance: self.linear_tolerance,
            line_search: self.line_search,
            max_backtracks: self.max_line_search_backtracks,
            backtrack_reduction: self.line_search_reduction,
            max_cutbacks: self.max_load_cutbacks,
            max_accepted_subincrements: self.max_accepted_increments,
            minimum_load_increment: self.minimum_load_increment,
        }
    }

    pub fn validate_controls(&self) -> Result<(), crate::FeaRunError> {
        super::NonlinearWorkLimits::derive(self)?;
        self.newton_policy()
            .validate()
            .map_err(|e| crate::FeaRunError::InvalidModel(e.to_string()))
    }
}
