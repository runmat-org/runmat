use runmat_meshing_core::contracts::ProtectedBoundaryComplex;

use super::{
    generate_convex_polyhedron_tetrahedron_mesh_from_plc,
    generate_single_tetrahedron_mesh_from_plc, generate_structured_box_tetrahedron_mesh_from_plc,
    TetrahedronGenerationError, TetrahedronMesh,
};

pub fn generate_solver_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    match generate_structured_box_tetrahedron_mesh_from_plc(plc) {
        Ok(mesh) => Ok(mesh),
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc) => {
            match generate_single_tetrahedron_mesh_from_plc(plc) {
                Ok(mesh) => Ok(mesh),
                Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc) => {
                    generate_convex_polyhedron_tetrahedron_mesh_from_plc(plc)
                }
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}
