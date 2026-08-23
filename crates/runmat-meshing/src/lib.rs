//! Topology-first meshing pipeline orchestration.

pub mod visualization;
pub use visualization::{
    map_nodal_vector_field_to_boundary_faces, map_nodal_vector_field_to_boundary_nodes,
    map_volume_scalar_field_to_boundary_faces, BoundaryFaceScalarValue, BoundaryFaceVectorValue,
    BoundaryNodeVectorValue, FieldMappingError,
};

#[cfg(test)]
mod structure_tests;
