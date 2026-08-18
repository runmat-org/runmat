mod batch;
mod batch_codec;
mod join;
mod surface_codec;
mod types;

pub use batch::{
    build_exact_face_mesh_batch, face_partition_descriptors, validate_exact_face_mesh_batch,
};
#[cfg(test)]
pub(crate) use batch_codec::decode_with_byte_limit;
pub use batch_codec::{decode_exact_face_mesh_batch, encode_exact_face_mesh_batch};
pub use join::{join_exact_face_mesh_batches, validate_exact_surface_mesh};
#[cfg(test)]
pub(crate) use surface_codec::decode_exact_surface_mesh_with_byte_limit;
pub use surface_codec::{decode_exact_surface_mesh, encode_exact_surface_mesh};
pub use types::{
    ExactFaceMeshBatch, ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError,
    ExactSurfaceMeshErrorKind, ExactSurfaceShellEvidence, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
    EXACT_SURFACE_MESH_SCHEMA_VERSION,
};
