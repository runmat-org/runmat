//! Physical static vector-potential curl-curl formulation with a discrete weak
//! Coulomb gauge. No reduced graph, penalty gauge, or fictitious frequency.
mod assembly;
mod quantity_layout;
mod boundary;
mod input;
mod plan;
pub(crate) use plan::DerivedMagneticSystem;
pub use plan::MagneticSystemPlan;
pub(crate) mod resources;
mod response;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

use crate::topology::AnalysisProjection;
pub(crate) use input::{compile_canonical_input, CompiledMagneticInput};
pub(crate) fn validate_projection_envelope(
    projection: &AnalysisProjection,
    grounded_faces: &[StableDigest],
) -> Result<crate::topology::TetrahedralEdgeSpace, MagnetostaticError> {
    Ok(validated_spaces(projection, grounded_faces)?.0)
}
/// Every exterior face of the projection, the fully grounded boundary.
pub fn exterior_face_identities(projection: &AnalysisProjection) -> Vec<StableDigest> {
    let mut faces: Vec<_> = projection
        .boundary_faces
        .iter()
        .filter(|face| face.role == runmat_meshing_core::BoundaryFaceRole::Exterior)
        .map(|face| face.stable_identity)
        .collect();
    faces.sort_unstable();
    faces
}
fn validated_spaces(
    projection: &AnalysisProjection,
    grounded_faces: &[StableDigest],
) -> Result<
    (
        crate::topology::TetrahedralEdgeSpace,
        boundary::MagneticBoundarySpace,
    ),
    MagnetostaticError,
> {
    let edges = crate::topology::TetrahedralEdgeSpace::derive(projection)
        .map_err(|e| MagnetostaticError::Topology(e.to_string()))?;
    let boundary = boundary::MagneticBoundarySpace::derive(projection, &edges, grounded_faces)?;
    assembly::validate_affine_geometry(projection)?;
    Ok((edges, boundary))
}
pub use input::{solve_canonical_static, validate_canonical_input};
pub use response::{
    element_mean, MagneticFieldSample, MagnetostaticResponse, MAGNETIC_SAMPLES_PER_ELEMENT,
};

use runmat_meshing_core::StableDigest;

/// One grounded exterior face held at a known tangential magnetic vector potential (Wb/m).
pub type PrescribedVectorPotential = (
    StableDigest,
    runmat_types::physical::Quantity<
        runmat_types::physical::kind::MagneticVectorPotential,
        [f64; 3],
    >,
);
/// CODATA 2022 vacuum permeability, SI H/m (not an exact SI defining constant).
pub use runmat_types::physical::VACUUM_MAGNETIC_PERMEABILITY as VACUUM_PERMEABILITY_H_PER_M;
use thiserror::Error;

/// Constant isotropic element law and physical volume current in SI units.
/// Identity is canonical, not vector position or an authored selection alias.
#[derive(Debug, Clone, Copy)]
pub struct MagneticVolumeData {
    pub element_identity: StableDigest,
    pub permeability_h_per_m: runmat_types::physical::MagneticPermeability,
    pub current_density_a_per_m2: runmat_types::physical::Quantity<
        runmat_types::physical::kind::ElectricCurrentDensity,
        [f64; 3],
    >,
}

#[derive(Debug, Error)]
pub enum MagnetostaticError {
    #[error("magnetic assembly numerical accounting failed: {0}")]
    Accounting(runmat_simulation_usage::LedgerError),
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

/// Numerical kernel with tangential A = 0 on `grounded_faces` (one connected exterior
/// surface) and magnetic insulation (n x H = 0) on every other exterior face. This does not
/// infer authored boundary policy: authored execution compiles the grounds before calling it.
/// Lowest-order edges are used on affine Tet4/Tet10 geometry.
pub fn solve_grounded_volume(
    projection: &AnalysisProjection,
    grounded_faces: &[StableDigest],
    prescribed_potential: &[PrescribedVectorPotential],
    volumes: &[MagneticVolumeData],
    residual_target: f64,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MagnetostaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    response::solve(
        assembly::MagneticAssembly::assemble(
            projection,
            grounded_faces,
            prescribed_potential,
            volumes,
        )?,
        residual_target,
    )
}

/// Execute the same kernel using exact precomputed sparse support.
/// The plan is revalidated against canonical topology before physical assembly;
/// it does not authorize scheduler placement, publication or production execution.
pub fn solve_grounded_volume_with_plan(
    projection: &AnalysisProjection,
    volumes: &[MagneticVolumeData],
    prescribed_potential: &[PrescribedVectorPotential],
    residual_target: f64,
    plan: &MagneticSystemPlan,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MagnetostaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    response::solve(
        assembly::MagneticAssembly::assemble_with_plan(
            projection,
            volumes,
            prescribed_potential,
            plan,
        )?,
        residual_target,
    )
}

/// Execute on a system derived in this process from `projection` (see
/// [`MagneticSystemPlan::derive_with_spaces`]); the topology is not derived again.
pub(crate) fn solve_grounded_volume_derived(
    projection: &AnalysisProjection,
    volumes: &[MagneticVolumeData],
    prescribed_potential: &[PrescribedVectorPotential],
    residual_target: f64,
    derived: DerivedMagneticSystem,
) -> Result<MagnetostaticResponse, MagnetostaticError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(MagnetostaticError::InvalidInput(
            "relative residual target".into(),
        ));
    }
    response::solve(
        assembly::MagneticAssembly::assemble_derived(
            projection,
            volumes,
            prescribed_potential,
            derived,
        )?,
        residual_target,
    )
}
