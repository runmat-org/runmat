pub mod boundary_smoothing;
pub mod interior_smoothing;

pub use boundary_smoothing::{
    smooth_tetrahedron_mesh_boundary_with_projector, TetrahedronBoundarySmoothingProjection,
    TetrahedronBoundarySmoothingProjector, TetrahedronMeshBoundarySmoothingOptions,
    TetrahedronMeshBoundarySmoothingReport,
};
pub use interior_smoothing::{
    smooth_tetrahedron_mesh_interior, TetrahedronMeshInteriorSmoothingOptions,
    TetrahedronMeshInteriorSmoothingReport,
};
