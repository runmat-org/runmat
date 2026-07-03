pub use runmat_meshing_core::contracts::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};

pub const MODULE_PURPOSE: &str = "deterministic Tetrahedron4 generation from a validated PLC";

mod convex_polyhedron;
mod initial;
mod single_tetrahedron;
mod solver;
mod structured_box;
mod types;
mod validation;
pub use convex_polyhedron::generate_convex_polyhedron_tetrahedron_mesh_from_plc;
pub use initial::generate_initial_tetrahedron_mesh_from_plc;
pub use single_tetrahedron::generate_single_tetrahedron_mesh_from_plc;
pub use solver::generate_solver_tetrahedron_mesh_from_plc;
pub use structured_box::generate_structured_box_tetrahedron_mesh_from_plc;
pub use types::TetrahedronGenerationError;

#[cfg(test)]
mod tests;
