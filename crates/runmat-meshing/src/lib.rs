//! Topology-first meshing pipeline orchestration.

pub mod solid;

pub use solid::{
    generate_analysis_mesh, generate_analysis_mesh_with_sizing, generate_solid_analysis_mesh,
    generate_solid_analysis_mesh_with_sizing, SolidMeshingError,
};
