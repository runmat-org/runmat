//! Deterministic logical-record chunking independent of execution placement and storage paths.

mod closure;
mod codec;
mod packing;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{MeshingChunkDescriptor, MeshingChunkMediaType, MeshingContractError, StableDigest};

pub use closure::{
    build_closed_stage_manifest, close_stage_payload_with_admission, decode_stage_manifest_streams,
    decode_stage_manifest_streams_borrowed, decode_stage_manifest_streams_borrowed_with_admission,
    decode_stage_manifest_streams_with_cancellation, verify_stage_manifest_closure,
    verify_stage_manifest_closure_borrowed, verify_stage_manifest_closure_borrowed_with_admission,
    ClosedMeshingStage, MeshingManifestMetadata, MeshingStreamStorageProfile,
};

const MAX_CHUNKS: usize = 65_536;
pub(super) const MAX_RECORDS_PER_CHUNK: usize = 1_000_000;
const MAX_CHUNK_BYTES: u64 = 512 * 1024 * 1024;
// Covers the fixed domain prefix and the worst-case definite envelope fields. Record byte-string
// framing is accounted separately by `encoded_bytes_cost`.
const CHUNK_OVERHEAD_RESERVE: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingChunkPolicy {
    pub maximum_chunk_bytes: u64,
    pub maximum_records_per_chunk: u32,
    pub maximum_total_encoded_bytes: u64,
}

impl MeshingChunkPolicy {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.maximum_chunk_bytes < 512
            || self.maximum_records_per_chunk == 0
            || self.maximum_records_per_chunk as usize > MAX_RECORDS_PER_CHUNK
            || self.maximum_chunk_bytes > MAX_CHUNK_BYTES
            || self.maximum_total_encoded_bytes < self.maximum_chunk_bytes
        {
            return Err(MeshingContractError::invalid(
                "meshing chunk policy",
                "chunk size must be at least 512 bytes and fit within non-zero record/total limits",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshingChunkStream {
    pub media_type: MeshingChunkMediaType,
    pub schema_version: u16,
    pub records: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedMeshingChunk {
    pub descriptor: MeshingChunkDescriptor,
    pub bytes: Vec<u8>,
}

/// Borrow encoded bytes during closure validation; ownership stays with storage.
#[derive(Clone, Copy)]
pub struct EncodedMeshingChunkRef<'a> {
    pub descriptor: &'a MeshingChunkDescriptor,
    pub bytes: &'a [u8],
}
impl<'a> From<&'a EncodedMeshingChunk> for EncodedMeshingChunkRef<'a> {
    fn from(chunk: &'a EncodedMeshingChunk) -> Self {
        Self {
            descriptor: &chunk.descriptor,
            bytes: &chunk.bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshingChunkedPayload {
    pub logical_content_digest: StableDigest,
    pub logical_entity_count: u64,
    pub chunks: Vec<EncodedMeshingChunk>,
    pub total_encoded_length: u64,
}

/// Requested chunk storage derived from the actual packing layout and encoder.
/// Input record buffers remain caller-owned and must be admitted separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshingChunkStorageProfile {
    pub chunk_count: usize,
    pub total_encoded_length: u64,
    /// Payload buffers and their typed chunk inventory; excludes allocator overhead.
    pub requested_storage_bytes: u64,
}

pub fn profile_chunked_stage_payload(
    streams: &[MeshingChunkStream],
    policy: MeshingChunkPolicy,
    cancellation: &dyn super::MeshingCancellationSignal,
) -> Result<MeshingChunkStorageProfile, MeshingContractError> {
    let mut profile = MeshingChunkStorageProfile {
        chunk_count: 0,
        total_encoded_length: 0,
        requested_storage_bytes: 0,
    };
    packing::visit(
        streams,
        policy,
        cancellation,
        |ordinal, first, stream, fragments| {
            let length = codec::encoded_length(
                ordinal,
                first,
                stream.media_type,
                stream.schema_version,
                fragments,
                cancellation,
            )? as u64;
            if length > policy.maximum_chunk_bytes {
                return Err(MeshingContractError::invalid(
                    "meshing chunks",
                    "encoded chunk exceeds the hard chunk byte limit",
                ));
            }
            profile.total_encoded_length = profile
                .total_encoded_length
                .checked_add(length)
                .ok_or_else(|| {
                    MeshingContractError::invalid("meshing chunks", "byte total overflow")
                })?;
            if profile.total_encoded_length > policy.maximum_total_encoded_bytes {
                return Err(MeshingContractError::invalid(
                    "meshing chunks",
                    format!("chunk inventory requires at least {} bytes, exceeding the hard total byte limit {}", profile.total_encoded_length, policy.maximum_total_encoded_bytes),
                ));
            }
            profile.chunk_count += 1;
            profile.requested_storage_bytes = profile
                .total_encoded_length
                .checked_add(
                    (profile.chunk_count as u64)
                        .checked_mul(std::mem::size_of::<EncodedMeshingChunk>() as u64)
                        .ok_or_else(|| {
                            MeshingContractError::invalid(
                                "meshing chunks",
                                "chunk storage overflow",
                            )
                        })?,
                )
                .ok_or_else(|| {
                    MeshingContractError::invalid("meshing chunks", "chunk storage overflow")
                })?;
            Ok(())
        },
    )?;
    Ok(profile)
}

pub fn build_chunked_stage_payload(
    streams: &[MeshingChunkStream],
    policy: MeshingChunkPolicy,
) -> Result<MeshingChunkedPayload, MeshingContractError> {
    build_chunked_stage_payload_with_cancellation(streams, policy, &super::NeverCancelled)
}

pub fn build_chunked_stage_payload_with_cancellation(
    streams: &[MeshingChunkStream],
    policy: MeshingChunkPolicy,
    cancellation: &dyn super::MeshingCancellationSignal,
) -> Result<MeshingChunkedPayload, MeshingContractError> {
    // Fail the complete hard byte limit before allocating any encoded payload.
    let profile = profile_chunked_stage_payload(streams, policy, cancellation)?;
    let logical_content_digest = logical_content_digest_with_cancellation(streams, cancellation)?;
    let logical_entity_count = streams.iter().try_fold(0_u64, |count, stream| {
        count
            .checked_add(stream.records.len() as u64)
            .ok_or_else(|| MeshingContractError::invalid("meshing chunks", "entity count overflow"))
    })?;
    let mut chunks = Vec::with_capacity(profile.chunk_count);
    packing::visit(
        streams,
        policy,
        cancellation,
        |ordinal, first, stream, fragments| {
            chunks.push(codec::encode_chunk(
                ordinal,
                first,
                stream.media_type,
                stream.schema_version,
                fragments,
                cancellation,
            )?);
            Ok(())
        },
    )?;
    Ok(MeshingChunkedPayload {
        logical_content_digest,
        logical_entity_count,
        chunks,
        total_encoded_length: profile.total_encoded_length,
    })
}

fn validate_streams(streams: &[MeshingChunkStream]) -> Result<(), MeshingContractError> {
    if streams.is_empty()
        || !streams
            .windows(2)
            .all(|pair| pair[0].media_type < pair[1].media_type)
    {
        return Err(MeshingContractError::invalid(
            "meshing chunk streams",
            "streams must be non-empty, unique, and ordered by media type",
        ));
    }
    for stream in streams {
        if stream.schema_version == 0 || stream.records.is_empty() {
            return Err(MeshingContractError::invalid(
                "meshing chunk stream",
                "schema version and record collection must be non-empty",
            ));
        }
        if stream.records.iter().any(Vec::is_empty) {
            return Err(MeshingContractError::invalid(
                "meshing chunk record",
                "logical records must not be empty",
            ));
        }
    }
    Ok(())
}

fn logical_content_digest_with_cancellation(
    streams: &[MeshingChunkStream],
    cancellation: &dyn super::MeshingCancellationSignal,
) -> Result<StableDigest, MeshingContractError> {
    let record_count = streams.iter().try_fold(0_u64, |count, stream| {
        count
            .checked_add(stream.records.len() as u64)
            .ok_or_else(|| {
                MeshingContractError::invalid("logical content identity", "record count overflow")
            })
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"runmat-meshing-logical-content/v1\0");
    hasher.update(record_count.to_be_bytes());
    let mut ordinal = 0_u64;
    for stream in streams {
        cancellation_checkpoint(cancellation)?;
        let media = stream.media_type.media_type().as_bytes();
        for record in &stream.records {
            cancellation_checkpoint(cancellation)?;
            hasher.update(ordinal.to_be_bytes());
            hasher.update((media.len() as u32).to_be_bytes());
            hasher.update(media);
            hasher.update(stream.schema_version.to_be_bytes());
            hasher.update((record.len() as u64).to_be_bytes());
            for block in record.chunks(1024 * 1024) {
                cancellation_checkpoint(cancellation)?;
                hasher.update(block);
            }
            ordinal += 1;
        }
    }
    Ok(StableDigest::from_bytes(hasher.finalize().into()))
}

fn cancellation_checkpoint(
    cancellation: &dyn super::MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    if cancellation.is_cancelled() {
        return Err(MeshingContractError::invalid(
            "meshing chunk processing",
            "operation was cancelled",
        ));
    }
    Ok(())
}

fn encoded_fragment_cost(length: usize) -> usize {
    let header = match length {
        0..=23 => 1,
        24..=0xff => 2,
        0x100..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    };
    2 + header + length
}
