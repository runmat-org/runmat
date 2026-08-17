//! Deterministic logical-record chunking independent of execution placement and storage paths.

mod closure;
mod codec;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use super::{MeshingChunkDescriptor, MeshingChunkMediaType, MeshingContractError, StableDigest};

pub use closure::{
    build_closed_stage_manifest, decode_stage_manifest_streams, verify_stage_manifest_closure,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshingChunkedPayload {
    pub logical_content_digest: StableDigest,
    pub logical_entity_count: u64,
    pub chunks: Vec<EncodedMeshingChunk>,
    pub total_encoded_length: u64,
}

pub fn build_chunked_stage_payload(
    streams: &[MeshingChunkStream],
    policy: MeshingChunkPolicy,
) -> Result<MeshingChunkedPayload, MeshingContractError> {
    policy.validate()?;
    validate_streams(streams)?;
    let logical_content_digest = logical_content_digest(streams)?;
    let logical_entity_count = streams.iter().try_fold(0_u64, |count, stream| {
        count
            .checked_add(stream.records.len() as u64)
            .ok_or_else(|| MeshingContractError::invalid("meshing chunks", "entity count overflow"))
    })?;

    let mut chunks = Vec::new();
    let mut global_entity_ordinal = 0_u64;
    let payload_budget = usize::try_from(policy.maximum_chunk_bytes)
        .unwrap_or(usize::MAX)
        .saturating_sub(CHUNK_OVERHEAD_RESERVE);
    for stream in streams {
        let mut start = 0;
        while start < stream.records.len() {
            if chunks.len() >= MAX_CHUNKS {
                return Err(MeshingContractError::invalid(
                    "meshing chunks",
                    "chunk count exceeds the manifest limit",
                ));
            }
            let mut end = start;
            let mut payload_bytes = 0_usize;
            while end < stream.records.len()
                && end - start < policy.maximum_records_per_chunk as usize
            {
                let cost = encoded_bytes_cost(stream.records[end].len());
                if cost > payload_budget {
                    return Err(MeshingContractError::invalid(
                        "meshing chunk record",
                        "one logical record exceeds the chunk byte limit",
                    ));
                }
                if end > start && payload_bytes.saturating_add(cost) > payload_budget {
                    break;
                }
                payload_bytes = payload_bytes.checked_add(cost).ok_or_else(|| {
                    MeshingContractError::invalid("meshing chunks", "chunk size overflow")
                })?;
                end += 1;
            }
            let chunk = codec::encode_chunk(
                chunks.len() as u32,
                global_entity_ordinal,
                stream.media_type,
                stream.schema_version,
                &stream.records[start..end],
            )?;
            if chunk.bytes.len() as u64 > policy.maximum_chunk_bytes {
                return Err(MeshingContractError::invalid(
                    "meshing chunks",
                    "encoded chunk exceeds the hard chunk byte limit",
                ));
            }
            global_entity_ordinal = global_entity_ordinal
                .checked_add((end - start) as u64)
                .ok_or_else(|| {
                    MeshingContractError::invalid("meshing chunks", "entity ordinal overflow")
                })?;
            chunks.push(chunk);
            start = end;
        }
    }

    let total_encoded_length = chunks.iter().try_fold(0_u64, |total, chunk| {
        total
            .checked_add(chunk.descriptor.encoded_length)
            .ok_or_else(|| MeshingContractError::invalid("meshing chunks", "byte total overflow"))
    })?;
    if total_encoded_length > policy.maximum_total_encoded_bytes {
        return Err(MeshingContractError::invalid(
            "meshing chunks",
            "chunk inventory exceeds the hard total byte limit",
        ));
    }
    Ok(MeshingChunkedPayload {
        logical_content_digest,
        logical_entity_count,
        chunks,
        total_encoded_length,
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

fn logical_content_digest(
    streams: &[MeshingChunkStream],
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
        let media = stream.media_type.media_type().as_bytes();
        for record in &stream.records {
            hasher.update(ordinal.to_be_bytes());
            hasher.update((media.len() as u32).to_be_bytes());
            hasher.update(media);
            hasher.update(stream.schema_version.to_be_bytes());
            hasher.update((record.len() as u64).to_be_bytes());
            hasher.update(record);
            ordinal += 1;
        }
    }
    Ok(StableDigest::from_bytes(hasher.finalize().into()))
}

fn encoded_bytes_cost(length: usize) -> usize {
    let header = match length {
        0..=23 => 1,
        24..=0xff => 2,
        0x100..=0xffff => 3,
        0x1_0000..=0xffff_ffff => 5,
        _ => 9,
    };
    header + length
}
