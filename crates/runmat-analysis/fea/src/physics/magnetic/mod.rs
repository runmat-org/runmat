//! Physical static vector-potential curl-curl formulation with a discrete weak
//! Coulomb gauge. No reduced graph, penalty gauge, or fictitious frequency.
mod assembly;
mod boundary;
mod input;
mod resources;
mod response;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

use crate::topology::AnalysisProjection;
pub(crate) use input::compile_canonical_input;
pub(crate) fn validate_projection_envelope(
    projection: &AnalysisProjection,
) -> Result<crate::topology::TetrahedralEdgeSpace, MagnetostaticError> {
    let edges = crate::topology::TetrahedralEdgeSpace::derive(projection)
        .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
    boundary::MagneticBoundarySpace::derive(projection, &edges)?;
    assembly::validate_affine_geometry(projection)?;
    Ok(edges)
}
pub use input::{solve_canonical_static, validate_canonical_input};
pub use response::{MagneticFieldSample, MagnetostaticResponse};

/// CODATA 2022 vacuum permeability, SI H/m (not an exact SI defining constant).
pub const VACUUM_PERMEABILITY_H_PER_M: f64 = 1.256_637_061_27e-6;
use runmat_meshing_core::StableDigest;
use thiserror::Error;

/// Constant isotropic element law and physical volume current in SI units.
/// Identity is canonical, not vector position or an authored selection alias.
#[derive(Debug, Clone, Copy)]
pub struct MagneticVolumeData {
    pub element_identity: StableDigest,
    pub permeability_h_per_m: f64,
    pub current_density_a_per_m2: [f64; 3],
}

#[derive(Debug, Error)]
pub enum MagnetostaticError {
    #[error("invalid magnetic input: {0}")]
    InvalidInput(String),
    #[error("unsupported magnetic envelope: {0}")]
    Unsupported(String),
    #[error("invalid magnetic topology: {0}")]
    Topology(String),
    #[error("invalid magnetic operator: {0}")]
    Operator(String),
    #[error("magnetic current is incompatible with the discrete gradient kernel (defect {0:e})")]
    IncompatibleCurrent(f64),
    #[error("magnetic solve failed acceptance: {0}")]
    NotConverged(String),
}

/// Experimental numerical kernel with explicitly full-exterior tangential A=0.
/// This does not silently infer authored boundary policy: authored execution must
/// separately compile and prove this boundary/gauge contract before calling it.
/// Lowest-order edges are used on affine Tet4/Tet10 geometry. General curved
/// geometry remains closed until rational form quadrature convergence is governed.
/// Production, native lifecycle, resource admission and restart are not advertised.
pub fn solve_full_grounded_volume(
    projection: &AnalysisProjection,
    volumes: &[MagneticVolumeData],
    residual_target: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MagnetostaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    response::solve(
        assembly::MagneticAssembly::assemble(projection, volumes)?,
        residual_target,
    )
}
