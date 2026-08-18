use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;

use super::{
    validate_exact_face_mesh_batch, ExactFaceMeshBatch, ExactSurfaceMeshError,
    ExactSurfaceMeshErrorKind,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-face-batch-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-face-mesh-batch/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(256 * 1024 * 1024, 10_000_000, 1024 * 1024, 64);

pub fn encode_exact_face_mesh_batch(
    batch: &ExactFaceMeshBatch,
    topology: &ExactBRepTopology,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_face_mesh_batch(batch, topology)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, batch, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

pub fn decode_exact_face_mesh_batch(
    bytes: &[u8],
    topology: &ExactBRepTopology,
) -> Result<ExactFaceMeshBatch, ExactSurfaceMeshError> {
    decode_with_limits(bytes, topology, CONTRACT_LIMITS)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    limits: CanonicalLimits,
) -> Result<ExactFaceMeshBatch, ExactSurfaceMeshError> {
    let batch =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    validate_exact_face_mesh_batch(&batch, topology)?;
    Ok(batch)
}

#[cfg(test)]
pub(crate) fn decode_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    maximum_encoded_bytes: usize,
) -> Result<ExactFaceMeshBatch, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        CanonicalLimits::new(maximum_encoded_bytes, 10_000_000, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        ExactSurfaceMeshErrorKind::InvalidEncoding,
        format!("{}: {}", error.field, error.reason),
    )
}
