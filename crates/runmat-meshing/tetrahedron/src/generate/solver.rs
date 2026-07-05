use runmat_meshing_core::contracts::ProtectedBoundaryComplex;

use super::{
    generate_convex_polyhedron_tetrahedron_mesh_from_plc,
    generate_nested_tetrahedron_shell_tetrahedron_mesh_from_plc,
    generate_single_tetrahedron_mesh_from_plc,
    generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc,
    generate_structured_box_tetrahedron_mesh_from_plc, TetrahedronGenerationError, TetrahedronMesh,
};

pub fn generate_solver_tetrahedron_mesh_from_plc(
    plc: &ProtectedBoundaryComplex,
) -> Result<TetrahedronMesh, TetrahedronGenerationError> {
    match generate_nested_tetrahedron_shell_tetrahedron_mesh_from_plc(plc) {
        Ok(mesh) => return Ok(mesh),
        Err(TetrahedronGenerationError::UnsupportedNestedTetrahedronShellPlc) => {}
        Err(err) => return Err(err),
    }
    match generate_structured_box_tetrahedron_mesh_from_plc(plc) {
        Ok(mesh) => Ok(mesh),
        Err(TetrahedronGenerationError::UnsupportedStructuredBoxPlc) => {
            match generate_single_tetrahedron_mesh_from_plc(plc) {
                Ok(mesh) => Ok(mesh),
                Err(TetrahedronGenerationError::UnsupportedSingleTetrahedronPlc) => {
                    match generate_convex_polyhedron_tetrahedron_mesh_from_plc(plc) {
                        Ok(mesh) => Ok(mesh),
                        Err(TetrahedronGenerationError::UnsupportedConvexPolyhedronPlc) => {
                            generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc(plc)
                        }
                        Err(err) => Err(err),
                    }
                }
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}
