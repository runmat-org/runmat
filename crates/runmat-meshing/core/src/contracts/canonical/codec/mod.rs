//! Bounded canonical serialization and domain-separated content identity.
//!
//! The encoding is a versioned, deterministic CBOR projection of the serde contract schema.
//! Object keys use UTF-8 byte order, all arrays retain their contract-defined order, all floating
//! point values use their exact finite `f64` representation, and decoders reject non-canonical
//! encodings. Type domains are embedded in the bytes, so equal-shaped contracts cannot collide.

mod contracts;

#[cfg(test)]
mod tests;

use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest as _, Sha256};

use super::{MeshingContractError, StableDigest};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-canonical-cbor/v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshingCanonicalLimits {
    pub maximum_encoded_bytes: usize,
    pub maximum_collection_items: usize,
    pub maximum_string_bytes: usize,
    pub maximum_nesting_depth: usize,
}

impl From<MeshingCanonicalLimits> for CanonicalLimits {
    fn from(value: MeshingCanonicalLimits) -> Self {
        Self::new(
            value.maximum_encoded_bytes,
            value.maximum_collection_items,
            value.maximum_string_bytes,
            value.maximum_nesting_depth,
        )
    }
}

impl MeshingCanonicalLimits {
    pub const IDENTITY: Self = Self::new(4 * 1024 * 1024, 65_536, 1024 * 1024, 64);
    pub const REQUEST: Self = Self::new(32 * 1024 * 1024, 100_000, 1024 * 1024, 64);
    pub const MANIFEST: Self = Self::new(64 * 1024 * 1024, 100_000, 1024 * 1024, 64);
    pub const ARTIFACT: Self = Self::new(512 * 1024 * 1024, 20_000_000, 8 * 1024 * 1024, 64);

    pub const fn new(
        maximum_encoded_bytes: usize,
        maximum_collection_items: usize,
        maximum_string_bytes: usize,
        maximum_nesting_depth: usize,
    ) -> Self {
        Self {
            maximum_encoded_bytes,
            maximum_collection_items,
            maximum_string_bytes,
            maximum_nesting_depth,
        }
    }
}

pub trait CanonicalMeshingContract: Serialize + DeserializeOwned + Sized {
    const DOMAIN: &'static str;
    const LIMITS: MeshingCanonicalLimits;

    fn validate_canonical(&self) -> Result<(), MeshingContractError>;

    fn canonical_encode(&self) -> Result<Vec<u8>, MeshingContractError> {
        self.validate_canonical()?;
        encode_contract(Self::DOMAIN, self, Self::LIMITS)
    }

    fn canonical_decode(bytes: &[u8]) -> Result<Self, MeshingContractError> {
        let decoded: Self = decode_contract(Self::DOMAIN, bytes, Self::LIMITS)?;
        decoded.validate_canonical()?;
        Ok(decoded)
    }

    fn canonical_digest(&self) -> Result<StableDigest, MeshingContractError> {
        let encoded = self.canonical_encode()?;
        Ok(StableDigest::from_bytes(Sha256::digest(encoded).into()))
    }
}

fn encode_contract<T: Serialize>(
    domain: &str,
    contract: &T,
    limits: MeshingCanonicalLimits,
) -> Result<Vec<u8>, MeshingContractError> {
    runmat_canonical_codec::encode_contract(CODEC_PREFIX, domain, contract, limits.into())
        .map_err(map_codec_error)
}

fn decode_contract<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: MeshingCanonicalLimits,
) -> Result<T, MeshingContractError> {
    runmat_canonical_codec::decode_contract(CODEC_PREFIX, domain, bytes, limits.into())
        .map_err(map_codec_error)
}

fn map_codec_error(error: CanonicalCodecError) -> MeshingContractError {
    MeshingContractError::invalid(error.field, error.reason)
}
