//! Continuum electric-phasor curl-curl formulation, exp(+i ω t).
//!
//! curl(μ⁻¹ curl E) − ω² ε E + i ω σ E = −i ω J, with full-exterior
//! homogeneous PEC (n × E = 0). Six first-kind lowest-order Nédélec corner
//! edges per affine Tet4/Tet10; Tet10 geometry is NOT quadratic electric-field
//! approximation. No gauge, graph surrogate, numerical damping or static limit.
//!
//! The initial uniqueness envelope requires strictly positive conductivity in
//! EVERY element. Nonnegative conductivity is a valid physical law, but zero-loss
//! regions/lossless cavity resonances remain unsupported until independent
//! nonresonance/coercivity authority exists. A small GMRES residual alone does
//! not prove uniqueness. Lossless cavities/waveguides are not discarded scope.
//!
//! This numerical owner does not qualify authored/native lifecycle, production,
//! resource admission, sweeps, transient Maxwell, open boundaries or restart.
//! Weak-form reference: https://aurora-multiphysics.github.io/apollo/frequency_domain/ComplexMaxwell.html
//! Basis/source reference: https://mfem.org/maxwell-notes/
mod assembly;
mod authored;
mod complex;
mod continuity;
mod geometry;
mod input;
mod plan;
mod recovery;
mod retained_consistency;
pub(crate) use retained_consistency::validate_retained_consistency;
pub(crate) mod resources;
mod response;
#[cfg(test)]
mod tests;

pub use authored::{compile_canonical_input, solve_canonical_harmonic, validate_canonical_input};
pub use complex::{ComplexPhasor, ComplexVector3};
pub use input::{MaxwellCurrentSample, MaxwellInput, MaxwellVolumeData};
pub use plan::MaxwellSystemPlan;
pub use response::{MaxwellFieldSample, MaxwellResponse};
use runmat_meshing_core::SolverMeshArtifact;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MaxwellError {
    #[error("invalid Maxwell input: {0}")]
    InvalidInput(String),
    #[error("unsupported Maxwell envelope: {0}")]
    Unsupported(String),
    #[error("invalid Maxwell topology: {0}")]
    Topology(String),
    #[error("invalid Maxwell operator: {0}")]
    Operator(String),
    #[error("Maxwell solve failed acceptance: {0}")]
    NotConverged(String),
}

/// Solve peak electric phasors with SI material coefficients and frozen source
/// values at every canonical Keast15 sample. No user callback executes here.
/// Tangential boundary E is exactly zero; no other field components are grounded.
pub fn solve_full_pec_volume(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    residual_target: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let plan = MaxwellSystemPlan::derive(mesh)?;
    solve_full_pec_volume_with_plan(mesh, input, residual_target, &plan)
}

/// Same physical solve with exact precomputed sparse support; the plan is checked
/// against canonical topology before quadrature, not trusted as solver evidence.
pub fn solve_full_pec_volume_with_plan(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    residual_target: f64,
    plan: &MaxwellSystemPlan,
) -> Result<MaxwellResponse, MaxwellError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MaxwellError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    response::solve(
        assembly::MaxwellAssembly::assemble(mesh, input, plan)?,
        residual_target,
    )
}

fn invalid(error: impl std::fmt::Display) -> MaxwellError {
    MaxwellError::InvalidInput(error.to_string())
}
fn topology(error: impl std::fmt::Display) -> MaxwellError {
    MaxwellError::Topology(error.to_string())
}
fn operator(error: impl std::fmt::Display) -> MaxwellError {
    MaxwellError::Operator(error.to_string())
}
