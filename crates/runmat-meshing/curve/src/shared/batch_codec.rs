use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;

use super::{batch::validate_batch, SharedCurveBatch, SharedCurveError};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-curve-batch-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "shared-curve-batch/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 2_500_000, 1024 * 1024, 64);

/// Encodes one independently admitted canonical edge batch.
pub fn encode_shared_curve_batch(
    batch: &SharedCurveBatch,
    topology: &ExactBRepTopology,
) -> Result<Vec<u8>, SharedCurveError> {
    validate_batch(batch, topology)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, batch, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

/// Decodes bounded hostile input and independently re-admits its topology slice.
pub fn decode_shared_curve_batch(
    bytes: &[u8],
    topology: &ExactBRepTopology,
) -> Result<SharedCurveBatch, SharedCurveError> {
    decode_with_limits(bytes, topology, CONTRACT_LIMITS)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    limits: CanonicalLimits,
) -> Result<SharedCurveBatch, SharedCurveError> {
    let batch: SharedCurveBatch =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    validate_batch(&batch, topology)?;
    Ok(batch)
}

#[cfg(test)]
pub(super) fn decode_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    maximum_encoded_bytes: usize,
) -> Result<SharedCurveBatch, SharedCurveError> {
    decode_with_limits(
        bytes,
        topology,
        CanonicalLimits::new(maximum_encoded_bytes, 2_500_000, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> SharedCurveError {
    SharedCurveError::invalid_encoding(error.field, error.reason)
}
