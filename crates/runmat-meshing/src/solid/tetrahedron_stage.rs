use runmat_meshing_core::contracts::ProtectedBoundaryComplex;
use runmat_meshing_tetrahedron::generate::{
    generate_solver_tetrahedron_mesh_from_plc, TetrahedronMesh,
};

use super::SolidMeshingError;

pub(super) fn generate_solid_tetrahedron_mesh(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, SolidMeshingError> {
    generate_solver_tetrahedron_mesh_from_plc(plc).map_err(SolidMeshingError::Tetrahedron)
}
