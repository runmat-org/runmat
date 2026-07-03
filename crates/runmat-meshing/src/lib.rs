//! Topology-first meshing pipeline orchestration.

pub mod analysis_prep;
pub mod solid;
pub mod visualization;

pub use analysis_prep::{
    prepare_geometry_for_analysis, ElementFamilyHint, MeshConnectivityClass, MeshingOptions,
    MeshingPrepResult, MeshingProfile, MeshingProvenance, MeshingQualityReport,
    PreparedMeshDescriptor, RegionMeshMapping,
};
pub use solid::{
    generate_analysis_mesh, generate_analysis_mesh_with_sizing, generate_solid_analysis_mesh,
    generate_solid_analysis_mesh_with_sizing, SolidMeshingError,
};
pub use visualization::{
    map_nodal_vector_field_to_boundary_faces, map_nodal_vector_field_to_boundary_nodes,
    map_volume_scalar_field_to_boundary_faces, BoundaryFaceScalarValue, BoundaryFaceVectorValue,
    BoundaryNodeVectorValue, FieldMappingError,
};
