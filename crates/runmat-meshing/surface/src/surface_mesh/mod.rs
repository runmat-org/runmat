mod artifact;
mod batch;
mod batch_codec;
mod convergence;
mod join;
mod partition_result;
mod pass_result;
mod surface_codec;
mod types;

pub use artifact::validate_published_exact_surface_mesh;
pub use batch::{
    build_exact_face_mesh_batch, face_partition_descriptors, validate_exact_face_mesh_batch,
};
pub(crate) use batch::{validate_exact_face_mesh_batch_parts, validate_face_partition_descriptor};
#[cfg(test)]
pub(crate) use batch_codec::decode_with_byte_limit;
pub use batch_codec::{decode_exact_face_mesh_batch, encode_exact_face_mesh_batch};
pub use convergence::{
    decide_exact_surface_pass, resolve_exact_surface_pass, ExactSurfaceConvergenceError,
    ExactSurfaceConvergenceErrorKind, ExactSurfaceConvergenceOutcome,
};
pub use join::{join_exact_face_mesh_batches, validate_exact_surface_mesh};
#[cfg(test)]
pub(crate) use partition_result::decode_exact_face_partition_result_with_byte_limit;
pub use partition_result::{
    build_exact_face_partition_result, decode_exact_face_partition_result,
    encode_exact_face_partition_result, validate_exact_face_partition_result,
};
pub(crate) use partition_result::{
    build_exact_face_partition_result_with_boundary,
    validate_exact_face_partition_result_with_boundary,
};
#[cfg(test)]
pub(crate) use pass_result::decode_exact_surface_pass_result_with_byte_limit;
pub use pass_result::{
    decode_exact_surface_pass_result, encode_decided_exact_surface_pass_result,
    encode_exact_surface_pass_result, validate_exact_surface_pass_result,
};
#[cfg(test)]
pub(crate) use surface_codec::decode_exact_surface_mesh_with_byte_limit;
pub use surface_codec::{
    decode_exact_surface_mesh, decode_exact_surface_mesh_from_pass,
    decode_published_exact_surface_mesh, encode_exact_surface_mesh,
    encode_exact_surface_mesh_from_pass,
};
pub use types::{
    ExactFaceMeshBatch, ExactFacePartitionOutcome, ExactFacePartitionResult,
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    ExactSurfacePassOutcome, ExactSurfacePassResult, ExactSurfaceShellEvidence,
    EXACT_FACE_MESH_BATCH_SCHEMA_VERSION, EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
    MAX_EXACT_FACE_PARTITIONS,
};
