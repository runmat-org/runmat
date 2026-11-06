//! Bounded canonical serialization and domain-separated content identity.
//!
//! The encoding is a versioned, deterministic CBOR projection of the serde contract schema.
//! Object keys use UTF-8 byte order, all arrays retain their contract-defined order, all floating
//! point values use their exact finite `f64` representation, and decoders reject non-canonical
//! encodings. Type domains are embedded in the bytes, so equal-shaped contracts cannot collide.

mod contracts;
mod solver_projection;

#[cfg(test)]
mod tests;

use runmat_canonical_codec::{CanonicalCodecControl, CanonicalCodecError, CanonicalLimits};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest as _, Sha256};

use super::{MeshingCancellationSignal, MeshingContractError, StableDigest};

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
    /// Control documents: identities, requests and manifests. Their size does not grow with
    /// the mesh, so these small fixed bounds refuse hostile input early.
    pub const IDENTITY: Self = Self::new(4 * 1024 * 1024, 65_536, 1024 * 1024, 64);
    pub const REQUEST: Self = Self::new(32 * 1024 * 1024, 100_000, 1024 * 1024, 64);
    pub const MANIFEST: Self = Self::new(64 * 1024 * 1024, 100_000, 1024 * 1024, 64);
    /// Run data: solver artifacts, projections, transfer maps and lineage grow with the mesh.
    /// The codec keeps only the platform allocation ceiling; each collection item occupies at
    /// least one encoded byte, so the byte ceiling also bounds every collection. The run's
    /// admitted budget gates the actual size where the bytes are produced or materialized
    /// (`canonical_decode_with_admission`, the execution budget scope). Grammar, canonical
    /// form, string and nesting bounds still refuse malformed input.
    pub const ARTIFACT: Self = Self::new(
        runmat_canonical_codec::PLATFORM_MAXIMUM_ENCODED_BYTES,
        runmat_canonical_codec::PLATFORM_MAXIMUM_ENCODED_BYTES,
        8 * 1024 * 1024,
        64,
    );

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

    /// Allocation-free borrowed profile; semantic validation remains a separate gate.
    fn canonical_encoding_profile(
        &self,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, MeshingContractError> {
        runmat_canonical_codec::profile_canonical_encoding(self, Self::LIMITS.into())
            .map_err(map_codec_error)
    }

    fn canonical_encode(&self) -> Result<Vec<u8>, MeshingContractError> {
        self.validate_canonical()?;
        encode_contract(Self::DOMAIN, self, Self::LIMITS)
    }

    /// Admit typed canonical materialization and the domain validator's declared
    /// scratch before either allocation. Grammar/content are always checked anew.
    fn canonical_decode_with_admission(
        bytes: &[u8],
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
        admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
        validation_scratch: impl FnOnce(&Self) -> Result<u64, MeshingContractError>,
    ) -> Result<Self, MeshingContractError> {
        metadata::decode(bytes, control, admit, validation_scratch)
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

/// Adapts progress probes and latches cancellation for one codec operation.
/// Callers can classify the returned error using the observed operation state,
/// without querying a potentially changed external signal after the failure.
pub struct MeshingCanonicalCodecControl<'a> {
    cancellation: &'a dyn MeshingCancellationSignal,
    observed_cancellation: std::sync::atomic::AtomicBool,
}
impl<'a> MeshingCanonicalCodecControl<'a> {
    pub fn new(cancellation: &'a dyn MeshingCancellationSignal) -> Self {
        Self {
            cancellation,
            observed_cancellation: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub fn observed_cancellation(&self) -> bool {
        self.observed_cancellation
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}
impl CanonicalCodecControl for MeshingCanonicalCodecControl<'_> {
    fn is_cancelled(&self) -> bool {
        if self.observed_cancellation() {
            return true;
        }
        let cancelled = self.cancellation.is_cancelled();
        if cancelled {
            self.observed_cancellation
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        cancelled
    }
}

pub(super) fn decode_contract_with_cancellation<T: DeserializeOwned>(
    domain: &str,
    bytes: &[u8],
    limits: MeshingCanonicalLimits,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<T, MeshingContractError> {
    runmat_canonical_codec::decode_contract_with_control(
        CODEC_PREFIX,
        domain,
        bytes,
        limits.into(),
        &MeshingCanonicalCodecControl::new(cancellation),
    )
    .map_err(map_codec_error)
}

pub(super) fn encode_contract_with_cancellation<T: Serialize>(
    domain: &str,
    contract: &T,
    limits: MeshingCanonicalLimits,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, MeshingContractError> {
    runmat_canonical_codec::encode_contract_with_control(
        CODEC_PREFIX,
        domain,
        contract,
        limits.into(),
        &MeshingCanonicalCodecControl::new(cancellation),
    )
    .map_err(map_codec_error)
}

fn map_codec_error(error: CanonicalCodecError) -> MeshingContractError {
    MeshingContractError::invalid(error.field, error.reason)
}

/// Hash an already encoded payload under the same cooperative execution authority.
/// Block boundaries affect responsiveness only; content identity is unchanged.
pub(super) fn digest_with_cancellation(
    bytes: &[u8],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<StableDigest, MeshingContractError> {
    let mut digest = Sha256::new();
    for block in bytes.chunks(64 * 1024) {
        if cancellation.is_cancelled() {
            return Err(MeshingContractError::invalid(
                "canonical digest",
                "cancelled",
            ));
        }
        digest.update(block);
    }
    Ok(StableDigest::from_bytes(digest.finalize().into()))
}

mod metadata;
#[cfg(test)]
pub(crate) use metadata::topology_decode_admission;

pub(super) use metadata::decode_validated;
pub(super) fn digest_borrowed_contract<T: Serialize>(
    domain: &str,
    value: &T,
    limits: MeshingCanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<StableDigest, MeshingContractError> {
    let bytes = runmat_canonical_codec::encode_contract_with_control(
        CODEC_PREFIX,
        domain,
        value,
        limits.into(),
        control,
    )
    .map_err(map_codec_error)?;
    Ok(StableDigest::from_bytes(Sha256::digest(bytes).into()))
}
