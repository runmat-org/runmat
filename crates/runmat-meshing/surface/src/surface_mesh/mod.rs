mod batch;
mod batch_codec;
mod types;

pub use batch::{
    build_exact_face_mesh_batch, face_partition_descriptors, validate_exact_face_mesh_batch,
};
#[cfg(test)]
pub(crate) use batch_codec::decode_with_byte_limit;
pub use batch_codec::{decode_exact_face_mesh_batch, encode_exact_face_mesh_batch};
pub use types::{
    ExactFaceMeshBatch, ExactSurfaceMesh, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    ExactSurfaceShellEvidence, EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
    EXACT_SURFACE_MESH_SCHEMA_VERSION,
};
