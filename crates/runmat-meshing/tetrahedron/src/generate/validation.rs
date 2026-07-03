use runmat_meshing_core::contracts::ProtectedBoundaryComplex;
use runmat_meshing_plc::validate::validate_protected_boundary_complex;

use super::TetrahedronGenerationError;

pub(super) fn validate_tetrahedron_generation_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<(), TetrahedronGenerationError> {
    validate_protected_boundary_complex(plc)
        .map(|_| ())
        .map_err(|error| TetrahedronGenerationError::InvalidProtectedBoundaryComplex { error })
}
