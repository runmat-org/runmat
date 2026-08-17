//! Bounded canonical serialization and domain-separated content identity.
//!
//! The encoding is a versioned, deterministic CBOR projection of the serde contract schema.
//! Object keys use UTF-8 byte order, all arrays retain their contract-defined order, all floating
//! point values use their exact finite `f64` representation, and decoders reject non-canonical
//! encodings. Type domains are embedded in the bytes, so equal-shaped contracts cannot collide.

mod contracts;
mod value;

#[cfg(test)]
mod tests;

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
    validate_domain(domain)?;
    let value = serde_json::to_value(contract)
        .map_err(|error| MeshingContractError::invalid("canonical encoding", error.to_string()))?;
    let mut encoded = Vec::with_capacity(1024);
    encoded.extend_from_slice(CODEC_PREFIX);
    let mut encoder = minicbor::Encoder::new(&mut encoded);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    value::encode_value(&mut encoder, &value)?;
    if encoded.len() > limits.maximum_encoded_bytes {
        return Err(MeshingContractError::invalid(
            "canonical encoding",
            "encoded contract exceeds its byte limit",
        ));
    }
    Ok(encoded)
}

fn decode_contract<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: MeshingCanonicalLimits,
) -> Result<T, MeshingContractError> {
    validate_domain(domain)?;
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            "encoded contract exceeds its byte limit",
        ));
    }
    let payload = bytes.strip_prefix(CODEC_PREFIX).ok_or_else(|| {
        MeshingContractError::invalid("canonical decoding", "codec domain prefix is missing")
    })?;
    let mut decoder = minicbor::Decoder::new(payload);
    require_array_length(decoder.array(), 2, "canonical envelope")?;
    let actual_domain = decoder.str().map_err(decoding_error)?;
    if actual_domain != domain {
        return Err(MeshingContractError::invalid(
            "canonical decoding domain",
            format!("expected {domain}, received {actual_domain}"),
        ));
    }
    let value = value::decode_value(&mut decoder, limits, 0)?;
    if decoder.position() != payload.len() {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            "trailing data is forbidden",
        ));
    }
    let canonical = encode_value_envelope(domain, &value)?;
    if canonical.as_slice() != bytes {
        return Err(MeshingContractError::invalid(
            "canonical decoding",
            "input is not in canonical form",
        ));
    }
    serde_json::from_value(value)
        .map_err(|error| MeshingContractError::invalid("canonical decoding", error.to_string()))
}

fn encode_value_envelope(
    domain: &str,
    value: &serde_json::Value,
) -> Result<Vec<u8>, MeshingContractError> {
    let mut encoded = CODEC_PREFIX.to_vec();
    let mut encoder = minicbor::Encoder::new(&mut encoded);
    encoder
        .array(2)
        .and_then(|encoder| encoder.str(domain))
        .map_err(encoding_error)?;
    value::encode_value(&mut encoder, value)?;
    Ok(encoded)
}

fn validate_domain(domain: &str) -> Result<(), MeshingContractError> {
    if domain.is_empty()
        || domain.len() > 128
        || !domain.is_ascii()
        || domain.chars().any(char::is_whitespace)
    {
        return Err(MeshingContractError::invalid(
            "canonical domain",
            "must be 1..=128 non-whitespace ASCII bytes",
        ));
    }
    Ok(())
}

fn require_array_length(
    length: Result<Option<u64>, minicbor::decode::Error>,
    expected: u64,
    field: &str,
) -> Result<(), MeshingContractError> {
    if length.map_err(decoding_error)? != Some(expected) {
        return Err(MeshingContractError::invalid(
            field,
            format!("expected a definite array of length {expected}"),
        ));
    }
    Ok(())
}

pub(super) fn encoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("canonical encoding", error.to_string())
}

pub(super) fn decoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("canonical decoding", error.to_string())
}
