pub mod field_mapping;

pub use field_mapping::{
    map_nodal_vector_field_to_boundary_faces, map_nodal_vector_field_to_boundary_nodes,
    map_volume_scalar_field_to_boundary_faces, BoundaryFaceScalarValue, BoundaryFaceVectorValue,
    BoundaryNodeVectorValue, FieldMappingError,
};
