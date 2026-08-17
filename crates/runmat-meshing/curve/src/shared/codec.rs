use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;

use super::{validation::invalid, SharedCurveMesh, SharedCurveValidationError};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-curve-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "shared-curve-mesh/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 1024 * 1024, 64);

/// Encodes an admitted shared-curve artifact in its sole canonical wire form.
pub fn encode_shared_curve_mesh(
    mesh: &SharedCurveMesh,
    topology: &ExactBRepTopology,
) -> Result<Vec<u8>, SharedCurveValidationError> {
    mesh.validate_against(topology)?;
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, mesh, CONTRACT_LIMITS)
        .map_err(map_codec_error)
}

/// Decodes bounded hostile input and independently re-admits it against exact topology.
pub fn decode_shared_curve_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
) -> Result<SharedCurveMesh, SharedCurveValidationError> {
    decode_with_limits(bytes, topology, CONTRACT_LIMITS)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    limits: CanonicalLimits,
) -> Result<SharedCurveMesh, SharedCurveValidationError> {
    let mesh: SharedCurveMesh =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?;
    mesh.validate_against(topology)?;
    Ok(mesh)
}

#[cfg(test)]
pub(super) fn decode_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    maximum_encoded_bytes: usize,
) -> Result<SharedCurveMesh, SharedCurveValidationError> {
    decode_with_limits(
        bytes,
        topology,
        CanonicalLimits::new(maximum_encoded_bytes, 20_000_000, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> SharedCurveValidationError {
    invalid(error.field, error.reason)
}
