use minicbor::{Decoder, Encoder};
use sha2::{Digest as _, Sha256};

use super::{
    EncodedMeshingChunk, MeshingChunkDescriptor, MeshingChunkMediaType, MAX_RECORDS_PER_CHUNK,
};
use crate::contracts::canonical::{MeshingContractError, StableDigest};

const CHUNK_PREFIX: &[u8] = b"runmat-meshing-logical-record-chunk/v2\0";
const CHUNK_ENVELOPE_SCHEMA_VERSION: u16 = 2;

pub(super) struct DecodedMeshingChunk {
    pub media_type: MeshingChunkMediaType,
    pub schema_version: u16,
    pub records: Vec<Vec<u8>>,
}

pub(super) fn encode_chunk(
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    records: &[Vec<u8>],
) -> Result<EncodedMeshingChunk, MeshingContractError> {
    let decoded_length = records.iter().try_fold(0_u64, |total, record| {
        total.checked_add(record.len() as u64).ok_or_else(|| {
            MeshingContractError::invalid("meshing chunk", "decoded length overflow")
        })
    })?;
    let bytes = encode_bytes(
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        records,
    )?;
    Ok(EncodedMeshingChunk {
        descriptor: MeshingChunkDescriptor {
            ordinal,
            first_logical_entity_ordinal,
            digest: StableDigest::from_bytes(Sha256::digest(&bytes).into()),
            media_type,
            schema_version,
            encoded_length: bytes.len() as u64,
            decoded_length,
            logical_entity_count: records.len() as u64,
        },
        bytes,
    })
}

pub(super) fn decode_chunk(
    chunk: &EncodedMeshingChunk,
) -> Result<DecodedMeshingChunk, MeshingContractError> {
    if chunk.descriptor.encoded_length != chunk.bytes.len() as u64
        || chunk.descriptor.digest != StableDigest::from_bytes(Sha256::digest(&chunk.bytes).into())
    {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "bytes do not match the content descriptor",
        ));
    }
    let payload = chunk.bytes.strip_prefix(CHUNK_PREFIX).ok_or_else(|| {
        MeshingContractError::invalid("meshing chunk", "chunk domain prefix is missing")
    })?;
    let mut decoder = Decoder::new(payload);
    if decoder.array().map_err(decoding_error)? != Some(6) {
        return Err(MeshingContractError::invalid(
            "meshing chunk",
            "expected a definite six-field envelope",
        ));
    }
    if decoder.u16().map_err(decoding_error)? != CHUNK_ENVELOPE_SCHEMA_VERSION {
        return Err(MeshingContractError::invalid(
            "meshing chunk",
            "unsupported chunk envelope version",
        ));
    }
    let media = decoder.str().map_err(decoding_error)?;
    let media_type = MeshingChunkMediaType::from_media_type(media).ok_or_else(|| {
        MeshingContractError::invalid("meshing chunk", "unknown meshing media type")
    })?;
    let schema_version = decoder.u16().map_err(decoding_error)?;
    let ordinal = decoder.u32().map_err(decoding_error)?;
    let first_logical_entity_ordinal = decoder.u64().map_err(decoding_error)?;
    let record_count = decoder.array().map_err(decoding_error)?.ok_or_else(|| {
        MeshingContractError::invalid("meshing chunk", "indefinite record arrays are forbidden")
    })?;
    if media_type != chunk.descriptor.media_type
        || schema_version != chunk.descriptor.schema_version
        || ordinal != chunk.descriptor.ordinal
        || first_logical_entity_ordinal != chunk.descriptor.first_logical_entity_ordinal
        || record_count != chunk.descriptor.logical_entity_count
    {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "chunk envelope does not match its descriptor",
        ));
    }
    let count = usize::try_from(record_count).map_err(|_| {
        MeshingContractError::invalid("meshing chunk", "record count does not fit in memory")
    })?;
    if count > MAX_RECORDS_PER_CHUNK || count > chunk.bytes.len() {
        return Err(MeshingContractError::invalid(
            "meshing chunk",
            "record count exceeds the bounded decoder limit",
        ));
    }
    let mut records = Vec::with_capacity(count);
    let mut decoded_length = 0_u64;
    for _ in 0..count {
        let record = decoder.bytes().map_err(decoding_error)?;
        decoded_length = decoded_length
            .checked_add(record.len() as u64)
            .ok_or_else(|| {
                MeshingContractError::invalid("meshing chunk", "decoded length overflow")
            })?;
        if decoded_length > chunk.descriptor.decoded_length {
            return Err(MeshingContractError::invalid(
                "meshing chunk closure",
                "decoded bytes exceed the descriptor bound",
            ));
        }
        records.push(record.to_vec());
    }
    if decoded_length != chunk.descriptor.decoded_length || decoder.position() != payload.len() {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "decoded length or trailing bytes violate the descriptor",
        ));
    }
    let canonical = encode_bytes(
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        &records,
    )?;
    if canonical != chunk.bytes {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "chunk bytes are not canonical",
        ));
    }
    Ok(DecodedMeshingChunk {
        media_type,
        schema_version,
        records,
    })
}

fn encode_bytes(
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    records: &[Vec<u8>],
) -> Result<Vec<u8>, MeshingContractError> {
    let mut bytes = CHUNK_PREFIX.to_vec();
    let mut encoder = Encoder::new(&mut bytes);
    encoder
        .array(6)
        .and_then(|encoder| encoder.u16(CHUNK_ENVELOPE_SCHEMA_VERSION))
        .and_then(|encoder| encoder.str(media_type.media_type()))
        .and_then(|encoder| encoder.u16(schema_version))
        .and_then(|encoder| encoder.u32(ordinal))
        .and_then(|encoder| encoder.u64(first_logical_entity_ordinal))
        .and_then(|encoder| encoder.array(records.len() as u64))
        .map_err(encoding_error)?;
    for record in records {
        encoder.bytes(record).map_err(encoding_error)?;
    }
    Ok(bytes)
}

fn encoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("meshing chunk encoding", error.to_string())
}

fn decoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("meshing chunk decoding", error.to_string())
}
