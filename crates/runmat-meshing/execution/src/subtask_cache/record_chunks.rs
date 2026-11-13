//! Purpose-neutral bounded record transport. The small manifest publishes last;
//! missing chunks are a cache miss. Numerical admission belongs to the consumer.
use super::*;
use runmat_geometry_core::GeometryAllocationReservation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub(super) const CHUNK: usize = 4 * 1024 * 1024;
pub(super) const MAXIMUM: usize = 1024 * 1024 * 1024;
#[derive(Clone, Copy)]
pub(super) struct Protocol {
    pub payload_domain: &'static [u8],
    pub manifest_key_domain: &'static [u8],
    pub prefix: &'static [u8],
    pub contract_domain: &'static str,
}

pub(super) const DIAGNOSTIC: Protocol = Protocol {
    payload_domain: b"runmat-meshing-diagnostic-chunks/v1\0",
    manifest_key_domain: b"runmat-meshing-diagnostic-manifest-key/v1\0",
    prefix: b"runmat-meshing-diagnostic-manifest/v1\0",
    contract_domain: "meshing-diagnostic-chunks/v1",
};

pub(super) const FACE: Protocol = Protocol {
    payload_domain: b"runmat-meshing-face-chunks/v1\0",
    manifest_key_domain: b"runmat-meshing-face-manifest-key/v1\0",
    prefix: b"runmat-meshing-face-manifest/v1\0",
    contract_domain: "meshing-face-chunks/v1",
};
pub(super) const LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(64 * 1024, 16_384, 4096, 16);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub(super) length: usize,
    pub(super) chunk_size: usize,
    pub(super) materialization: u64,
    pub(super) chunks: Vec<StableDigest>,
}

pub(super) fn digest(
    protocol: Protocol,
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    digest_in_domain(protocol.payload_domain, bytes, control)
}
pub(super) fn manifest_digest(
    protocol: Protocol,
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    digest_in_domain(protocol.manifest_key_domain, bytes, control)
}
fn digest_in_domain(
    domain: &[u8],
    bytes: &[u8],
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    let mut hash = Sha256::new();
    hash.update(domain);
    for chunk in bytes.chunks(64 * 1024) {
        control
            .checkpoint()
            .map_err(MeshingSubtaskCacheError::Control)?;
        control
            .consume_search_work(1)
            .map_err(MeshingSubtaskCacheError::Control)?;
        hash.update(chunk);
    }
    Ok(StableDigest::from_bytes(hash.finalize().into()))
}

pub(super) fn retain(
    protocol: Protocol,
    store: &dyn MeshingSubtaskCacheStore,
    bytes: &[u8],
    materialization: u64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<StableDigest>, MeshingSubtaskCacheError> {
    let chunk_size = usize::try_from(store.maximum_entry_bytes())
        .unwrap_or(usize::MAX)
        .min(CHUNK);
    if bytes.is_empty()
        || bytes.len() > MAXIMUM
        || chunk_size == 0
        || bytes.len().div_ceil(chunk_size) > 256
    {
        return Ok(None);
    }
    let count = bytes.len().div_ceil(chunk_size);
    let _metadata = GeometryAllocationReservation::new(
        control,
        (count * std::mem::size_of::<StableDigest>()) as u64,
    )
    .map_err(MeshingSubtaskCacheError::Control)?;
    let mut chunks = Vec::with_capacity(count);
    for chunk in bytes.chunks(chunk_size) {
        let key = digest(protocol, chunk, control)?;
        if !store.write(key, chunk, 0, control)? {
            return Ok(None);
        }
        chunks.push(key);
    }
    let codec = crate::codec_control::CodecControl::new(control);
    let manifest = Manifest {
        length: bytes.len(),
        chunk_size,
        materialization,
        chunks,
    };
    let encoding_profile =
        runmat_canonical_codec::profile_canonical_encoding_with_control(&manifest, LIMITS, &codec)
            .map_err(|error| codec.error(error))?;
    let manifest_materialization =
        runmat_canonical_codec::profile_serialized_with_control(&manifest, LIMITS, &codec)
            .map_err(|error| codec.error(error))?
            .materialization_resident_bytes_estimate;
    let _encoding_storage = GeometryAllocationReservation::new(
        control,
        encoding_profile.resident_workspace_bytes_estimate,
    )
    .map_err(MeshingSubtaskCacheError::Control)?;
    let encoded = runmat_canonical_codec::encode_contract_with_control(
        protocol.prefix,
        protocol.contract_domain,
        &manifest,
        LIMITS,
        &codec,
    )
    .map_err(|error| codec.error(error))?;
    let key = manifest_digest(protocol, &encoded, control)?;
    if !store.write(key, &encoded, manifest_materialization, control)? {
        return Ok(None);
    }
    Ok(Some(key))
}

/// Chunk authentication establishes only transport integrity. The supplied
/// domain inventory is recomputed before returning any typed materialization.
pub(super) fn read<'a>(
    protocol: Protocol,
    store: &dyn MeshingSubtaskCacheStore,
    key: StableDigest,
    control: &'a dyn GeometryEvaluationControl,
    inventory: impl FnOnce(&[u8]) -> Result<u64, MeshingSubtaskCacheError>,
) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
    let Some(entry) = store.read(key, control)? else {
        return Ok(None);
    };
    if entry.bytes.len() > LIMITS.maximum_encoded_bytes
        || entry.bytes.len() as u64 > store.maximum_entry_bytes()
        || manifest_digest(protocol, &entry.bytes, control)? != key
    {
        return Err(MeshingSubtaskCacheError::Storage(
            "chunk record manifest identity or bound differs".into(),
        ));
    }
    let codec = crate::codec_control::CodecControl::new(control);
    let manifest_materialization =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            std::iter::once(entry.bytes.as_slice()),
            &codec,
        )
        .map_err(|error| codec.error(error))?;
    if manifest_materialization > entry.materialization_resident_bytes {
        return Err(MeshingSubtaskCacheError::Storage(
            "chunk record manifest understates decoded storage".into(),
        ));
    }
    let manifest: Manifest = runmat_canonical_codec::decode_contract_with_control(
        protocol.prefix,
        protocol.contract_domain,
        &entry.bytes,
        LIMITS,
        &codec,
    )
    .map_err(|error| codec.error(error))?;
    if manifest.length == 0
        || manifest.length > MAXIMUM
        || manifest.chunk_size == 0
        || manifest.chunk_size > CHUNK
        || manifest.chunks.len() > 256
        || manifest.chunks.len() != manifest.length.div_ceil(manifest.chunk_size)
    {
        return Err(MeshingSubtaskCacheError::Storage(
            "chunk record chunk inventory differs".into(),
        ));
    }
    // Keep the manifest's admitted storage alive while its decoded chunk keys
    // coexist with the assembled payload and typed decoding reservation.
    let storage = GeometryAllocationReservation::new(
        control,
        (manifest.length as u64)
            .checked_add(manifest.materialization)
            .ok_or_else(|| {
                MeshingSubtaskCacheError::Storage("chunk record storage overflow".into())
            })?,
    )
    .map_err(MeshingSubtaskCacheError::Control)?;
    let mut bytes = Vec::with_capacity(manifest.length);
    for (index, key) in manifest.chunks.into_iter().enumerate() {
        let Some(entry) = store.read(key, control)? else {
            return Ok(None);
        };
        let expected = (manifest.length - index * manifest.chunk_size).min(manifest.chunk_size);
        if entry.bytes.len() != expected || digest(protocol, &entry.bytes, control)? != key {
            return Err(MeshingSubtaskCacheError::Storage(
                "chunk record chunk identity or length differs".into(),
            ));
        }
        bytes.extend_from_slice(&entry.bytes);
    }
    // Metadata is a claim; the domain owns the actual typed wire inventory.
    let required = inventory(&bytes)?;
    if required > manifest.materialization {
        return Err(MeshingSubtaskCacheError::Storage(
            "chunk record understates decoded storage".into(),
        ));
    }
    Ok(Some(MeshingSubtaskCacheEntry {
        bytes,
        storage,
        materialization_resident_bytes: manifest.materialization,
    }))
}
