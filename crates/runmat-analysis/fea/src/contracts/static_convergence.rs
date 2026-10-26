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
    /// How the operator was proved positive definite before the solve.
    pub operator_proof: StaticOperatorProof,
    /// PCG only: the Lanczos λ_min estimate and the forward error it implies. An estimate,
    /// never part of the acceptance criterion or the coercivity proof.
    pub spectral_estimate: Option<SpectralErrorEstimate>,
}

/// The forward error a PCG solve implies through its own Lanczos coefficients. PCG on `A` with
/// preconditioner `M` is Lanczos on `M⁻¹A`; the smallest Ritz value `θ_min` over-estimates
/// `λ_min(M⁻¹A)`, so `‖e‖_A ≤ √(rᵀM⁻¹r / λ_min)` is only estimated by `√(rᵀM⁻¹r / θ_min)`.
/// Labelled an estimate: it is never a bound and never admits a solve.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectralErrorEstimate {
    /// Smallest Ritz value of `M⁻¹A` (an over-estimate of λ_min).
    pub lambda_min_estimate: f64,
    /// Largest Ritz value of `M⁻¹A` (an under-estimate of λ_max).
    pub lambda_max_estimate: f64,
    /// Lanczos steps behind the estimate (iterations since the start or the last restart).
    pub lanczos_steps: u32,
    /// Flexible CG (a variable preconditioner, e.g. the device multigrid): the tridiagonal is
    /// only approximately Lanczos.
    pub flexible: bool,
    /// Estimated energy-norm error `√(rᵀM⁻¹r / θ_min)` of the returned solution.
    pub energy_error_estimate: f64,
    /// The estimate relative to the solution's energy norm `√(xᵀ(b − r))`; `None` when that norm
    /// is zero.
    pub relative_energy_error_estimate: Option<f64>,
}

impl SpectralErrorEstimate {
    fn is_valid(&self) -> bool {
        self.lambda_min_estimate.is_finite()
            && self.lambda_min_estimate > 0.0
            && self.lambda_max_estimate.is_finite()
            && self.lambda_max_estimate >= self.lambda_min_estimate
            && self.lanczos_steps > 0
            && self.energy_error_estimate.is_finite()
            && self.energy_error_estimate >= 0.0
            && self
                .relative_energy_error_estimate
                .is_none_or(|value| value.is_finite() && value >= 0.0)
    }
}

/// The coercivity proof that admitted a static solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticOperatorProof {
    /// Sparse Cholesky: every scaled pivot positive; the factor also solved.
    SparseCholesky,
    /// Element rigidity, face gluing and anchors (no global factor): positive definite in exact
    /// arithmetic without an eigenvalue bound. The solution is accepted by its true residual.
    ElasticRigidity,
}
impl StructuralStaticConvergence {
    pub(crate) fn from_solve(
        result: &crate::solve::linear::LinearSolveResult,
        operator_proof: StaticOperatorProof,
    ) -> Self {
        Self {
            operator_proof,
            spectral_estimate: result.spectral_estimate,
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
            || self
                .spectral_estimate
                .is_some_and(|estimate| !estimate.is_valid())
        {
            return Err(SolverFormulationCompatibilityError::InvalidStructuralStaticConvergence);
        }
        Ok(())
    }
}
