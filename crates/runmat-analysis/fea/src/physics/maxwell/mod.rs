//! Harmonic electromagnetic field formulation in E (equivalently A = E/(-i omega) in the
//! temporal gauge), exp(+i omega t):
//!
//! curl(μ⁻¹ curl E) − ω² ε E + i ω σ E = −i ω J,
//!
//! with lowest-order first-kind Nédélec corner edges per affine Tet4/Tet10 (Tet10 geometry is
//! not a quadratic field approximation). Boundary conditions: tangential E = 0 on grounded
//! exterior faces (`magnetic.vector_potential_ground`, perfect conductors) and the natural
//! condition n × H = 0 on every other exterior face. Sources: impressed volume current
//! density and terminal-driven solid conductors (current or voltage drive, see `drive`).
//! Conductivity, permittivity and permeability are per element; zero conductivity is
//! admitted (the operator keeps −ω²ε), so air and insulators need no artificial loss.
//! The augmented complex system is factored by sparse LU (`solve::sparse_direct`); acceptance
//! measures the original equations' residual, the complex Poynting balance and weak charge
//! continuity. No gauge, graph surrogate, numerical damping or static limit is added.
//! Weak-form reference: https://aurora-multiphysics.github.io/apollo/frequency_domain/ComplexMaxwell.html
//! Basis/source reference: https://mfem.org/maxwell-notes/
mod authored;
mod complex;
mod continuity;
mod drive;
mod geometry;
mod input;
mod plan;
pub(crate) mod resources;
mod response;
mod system;
pub(crate) use system::validate_retained_consistency;
#[cfg(test)]
pub(crate) mod tests;

pub use authored::{
    compile_canonical_input, grounded_face_identities, harmonic_frequencies,
    solve_canonical_harmonic, validate_canonical_input,
};
pub use complex::{ComplexPhasor, ComplexVector3};
pub use input::{
    MaxwellCurrentSample, MaxwellDriveExcitation, MaxwellElementVector, MaxwellInput,
    MaxwellTerminalDrive, MaxwellVolumeData,
};
pub use plan::MaxwellSystemPlan;
pub use response::{MaxwellFieldSample, MaxwellResponse};
use runmat_meshing_core::SolverMeshArtifact;
pub use system::MaxwellDriveResponse;
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

/// Solve one frequency of a compiled harmonic input.
pub fn solve(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    residual_target: f64,
) -> Result<MaxwellResponse, MaxwellError> {
    let plan = MaxwellSystemPlan::derive(mesh, &input.grounded_faces)?;
    solve_with_plan(mesh, input, residual_target, &plan)
}

/// Same solve with a precomputed plan; the plan is re-derived and compared, never trusted.
pub fn solve_with_plan(
    mesh: &SolverMeshArtifact,
    input: &MaxwellInput,
    residual_target: f64,
    plan: &MaxwellSystemPlan,
) -> Result<MaxwellResponse, MaxwellError> {
    system::solve(mesh, input, residual_target, plan)
}

fn invalid(error: impl std::fmt::Display) -> MaxwellError {
    MaxwellError::InvalidInput(error.to_string())
}
fn topology(error: impl std::fmt::Display) -> MaxwellError {
    MaxwellError::Topology(error.to_string())
}
