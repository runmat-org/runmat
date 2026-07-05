pub use runmat_meshing_core::contracts::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};

pub const MODULE_PURPOSE: &str = "deterministic Tetrahedron4 generation from a validated PLC";

mod convex_polyhedron;
mod evidence;
mod initial;
mod material;
mod single_tetrahedron;
mod solver;
mod star_shaped_polyhedron;
mod structured_box;
mod types;
mod validation;
pub(crate) use crate::protected_edges::source_edge_ids_for_face_edges;
pub use convex_polyhedron::generate_convex_polyhedron_tetrahedron_mesh_from_plc;
pub use initial::generate_initial_tetrahedron_mesh_from_plc;
pub use single_tetrahedron::generate_single_tetrahedron_mesh_from_plc;
pub use solver::generate_solver_tetrahedron_mesh_from_plc;
pub use star_shaped_polyhedron::generate_star_shaped_polyhedron_tetrahedron_mesh_from_plc;
pub use structured_box::generate_structured_box_tetrahedron_mesh_from_plc;
pub use types::TetrahedronGenerationError;

#[cfg(test)]
mod tests;
