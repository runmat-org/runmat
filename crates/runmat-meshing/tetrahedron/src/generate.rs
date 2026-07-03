pub use runmat_meshing_core::contracts::{
    Tetrahedron4Element, TetrahedronBoundaryFace, TetrahedronMesh, TetrahedronMeshNode,
};

pub const MODULE_PURPOSE: &str = "deterministic Tetrahedron4 generation from a validated PLC";

#[path = "generate/initial.rs"]
mod initial;
#[path = "generate/structured_box.rs"]
mod structured_box;
#[path = "generate/types.rs"]
mod types;
pub use initial::generate_initial_tetrahedron_mesh_from_plc;
pub use structured_box::generate_structured_box_tetrahedron_mesh_from_plc;
pub use types::TetrahedronGenerationError;

#[cfg(test)]
mod tests;
