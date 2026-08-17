use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest as _, Sha256};

use super::{GeometryContractError, GeometryDigest};

const GEOMETRY_CODEC_PREFIX: &[u8] = b"runmat-geometry-canonical-cbor/v1\0";

pub(crate) fn encode<T: Serialize>(
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, GeometryContractError> {
    runmat_canonical_codec::encode_contract(GEOMETRY_CODEC_PREFIX, domain, value, limits)
        .map_err(map_error)
}

pub(crate) fn decode<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
) -> Result<T, GeometryContractError> {
    runmat_canonical_codec::decode_contract(GEOMETRY_CODEC_PREFIX, domain, bytes, limits)
        .map_err(map_error)
}

pub(crate) fn digest(bytes: &[u8]) -> Result<GeometryDigest, GeometryContractError> {
    if bytes.is_empty() {
        return Err(GeometryContractError::invalid(
            "geometry component",
            "encoded component must not be empty",
        ));
    }
    Ok(GeometryDigest::from_bytes(Sha256::digest(bytes).into()))
}

fn map_error(error: CanonicalCodecError) -> GeometryContractError {
    GeometryContractError::invalid(error.field, error.reason)
}
