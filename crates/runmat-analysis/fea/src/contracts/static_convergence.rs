//! Original solver convergence, distinct from recovered physical force balance.
use super::SolverFormulationCompatibilityError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralStaticConvergence {
    pub converged: bool,
    pub iterations: u32,
    /// Absolute norm in the actual full, affine-reduced, or saddle-point equations.
    pub residual_norm: f64,
    /// Normalized by the original solver's reference norm, with no physical-load floor.
    pub relative_residual: Option<f64>,
}
impl StructuralStaticConvergence {
    pub(crate) fn from_solve(result: &crate::solve::linear::LinearSolveResult) -> Self {
        Self {
            converged: result.converged,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            relative_residual: result.relative_residual,
        }
    }
    pub fn validate(&self) -> Result<(), SolverFormulationCompatibilityError> {
        if !self.residual_norm.is_finite()
            || self.residual_norm < 0.0
            || self
                .relative_residual
                .is_some_and(|x| !x.is_finite() || x < 0.0)
            || (self.converged && self.relative_residual.is_none())
        {
            return Err(SolverFormulationCompatibilityError::InvalidStructuralStaticConvergence);
        }
        Ok(())
    }
}
