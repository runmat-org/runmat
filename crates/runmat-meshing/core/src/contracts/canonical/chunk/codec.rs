mod canonical;
use minicbor::{Decoder, Encoder};
use sha2::{Digest as _, Sha256};

use super::{
    EncodedMeshingChunk, MeshingChunkDescriptor, MeshingChunkMediaType, MAX_RECORDS_PER_CHUNK,
};
use crate::contracts::canonical::{MeshingCancellationSignal, MeshingContractError, StableDigest};

const CHUNK_PREFIX: &[u8] = b"runmat-meshing-logical-record-chunk/v3\0";
const CHUNK_ENVELOPE_SCHEMA_VERSION: u16 = 3;

#[derive(Clone, Copy)]
pub(super) struct MeshingRecordFragment<'a> {
    pub bytes: &'a [u8],
    pub completes_record: bool,
}

pub(super) struct BorrowedRecordFragment<'a> {
    pub bytes: &'a [u8],
    pub completes_record: bool,
}

pub(super) struct DecodedMeshingChunk<'a> {
    pub media_type: MeshingChunkMediaType,
    pub schema_version: u16,
    pub fragments: BorrowedFragments<'a>,
}

/// Borrow a validated definite fragment array without allocating an index.
#[derive(Clone, Copy)]
pub(super) struct BorrowedFragments<'a> {
    bytes: &'a [u8],
    count: usize,
}
impl<'a> BorrowedFragments<'a> {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn iter(&self) -> impl Iterator<Item = BorrowedRecordFragment<'a>> + '_ {
        let mut decoder = Decoder::new(self.bytes);
        (0..self.count).map(move |_| {
            decoder.array().expect("validated fragment array");
            let completes_record = decoder.bool().expect("validated fragment flag");
            let bytes = decoder.bytes().expect("validated fragment bytes");
            BorrowedRecordFragment {
                bytes,
                completes_record,
            }
        })
    }
}
pub(super) fn encode_chunk(
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    fragments: super::packing::Fragments<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<EncodedMeshingChunk, MeshingContractError> {
    let decoded_length = fragments.iter().try_fold(0_u64, |total, fragment| {
        total
            .checked_add(fragment.bytes.len() as u64)
            .ok_or_else(|| {
                MeshingContractError::invalid("meshing chunk", "decoded length overflow")
            })
    })?;
    let bytes = encode_bytes_with_cancellation(
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        fragments,
        cancellation,
    )?;
    let mut hasher = Sha256::new();
    for block in bytes.chunks(1024 * 1024) {
        checkpoint(cancellation)?;
        hasher.update(block);
    }
    Ok(EncodedMeshingChunk {
        descriptor: MeshingChunkDescriptor {
            ordinal,
            first_logical_entity_ordinal,
            digest: StableDigest::from_bytes(hasher.finalize().into()),
            media_type,
            schema_version,
            encoded_length: bytes.len() as u64,
            decoded_length,
            logical_entity_count: fragments
                .iter()
                .filter(|fragment| fragment.completes_record)
                .count() as u64,
        },
        bytes,
    })
}

pub(super) fn decode_chunk_with_cancellation<'a>(
    chunk: super::EncodedMeshingChunkRef<'a>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DecodedMeshingChunk<'a>, MeshingContractError> {
    checkpoint(cancellation)?;
    if chunk.descriptor.encoded_length != chunk.bytes.len() as u64 {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "bytes do not match the content descriptor",
        ));
    }
    let mut hasher = Sha256::new();
    for block in chunk.bytes.chunks(1024 * 1024) {
        checkpoint(cancellation)?;
        hasher.update(block);
    }
    if chunk.descriptor.digest != StableDigest::from_bytes(hasher.finalize().into()) {
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
    let fragment_count = decoder.array().map_err(decoding_error)?.ok_or_else(|| {
        MeshingContractError::invalid("meshing chunk", "indefinite fragment arrays are forbidden")
    })?;
    if media_type != chunk.descriptor.media_type
        || schema_version != chunk.descriptor.schema_version
        || ordinal != chunk.descriptor.ordinal
        || first_logical_entity_ordinal != chunk.descriptor.first_logical_entity_ordinal
    {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "chunk envelope does not match its descriptor",
        ));
    }
    let count = usize::try_from(fragment_count).map_err(|_| {
        MeshingContractError::invalid("meshing chunk", "fragment count does not fit in memory")
    })?;
    if count == 0 || count > MAX_RECORDS_PER_CHUNK || count > chunk.bytes.len() {
        return Err(MeshingContractError::invalid(
            "meshing chunk",
            "fragment count exceeds the bounded decoder limit",
        ));
    }
    let fragment_start = decoder.position();
    let mut completed_records = 0_u64;
    let mut decoded_length = 0_u64;
    for _ in 0..count {
        checkpoint(cancellation)?;
        if decoder.array().map_err(decoding_error)? != Some(2) {
            return Err(MeshingContractError::invalid(
                "meshing chunk",
                "expected a definite two-field record fragment",
            ));
        }
        let completes_record = decoder.bool().map_err(decoding_error)?;
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
        completed_records += u64::from(completes_record);
    }
    let fragments = BorrowedFragments {
        bytes: &payload[fragment_start..decoder.position()],
        count,
    };
    if decoded_length != chunk.descriptor.decoded_length
        || completed_records != chunk.descriptor.logical_entity_count
        || decoder.position() != payload.len()
    {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "decoded length or trailing bytes violate the descriptor",
        ));
    }
    // Re-encode into a comparing sink: canonicality needs no second payload.
    let mut canonical = canonical::CompareBytes::new(payload);
    let mut encoder = Encoder::new(&mut canonical);
    encoder
        .array(6)
        .and_then(|encoder| encoder.u16(CHUNK_ENVELOPE_SCHEMA_VERSION))
        .and_then(|encoder| encoder.str(media_type.media_type()))
        .and_then(|encoder| encoder.u16(schema_version))
        .and_then(|encoder| encoder.u32(ordinal))
        .and_then(|encoder| encoder.u64(first_logical_entity_ordinal))
        .and_then(|encoder| encoder.array(fragments.len() as u64))
        .map_err(encoding_error)?;
    for fragment in fragments.iter() {
        checkpoint(cancellation)?;
        encoder
            .array(2)
            .and_then(|encoder| encoder.bool(fragment.completes_record))
            .and_then(|encoder| encoder.bytes(fragment.bytes))
            .map_err(encoding_error)?;
    }
    if !canonical.complete() {
        return Err(MeshingContractError::invalid(
            "meshing chunk closure",
            "chunk bytes are not canonical",
        ));
    }
    Ok(DecodedMeshingChunk {
        media_type,
        schema_version,
        fragments,
    })
}

fn encode_bytes_with_cancellation(
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    fragments: super::packing::Fragments<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, MeshingContractError> {
    let length = encoded_length(
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        fragments,
        cancellation,
    )?;
    let mut bytes = Vec::with_capacity(length);
    bytes.extend_from_slice(CHUNK_PREFIX);
    let mut encoder = Encoder::new(&mut bytes);
    encode_payload(
        &mut encoder,
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        fragments,
        cancellation,
    )?;
    Ok(bytes)
}

pub(super) fn encoded_length(
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    fragments: super::packing::Fragments<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<usize, MeshingContractError> {
    let mut writer = canonical::CountBytes(CHUNK_PREFIX.len());
    encode_payload(
        &mut Encoder::new(&mut writer),
        ordinal,
        first_logical_entity_ordinal,
        media_type,
        schema_version,
        fragments,
        cancellation,
    )?;
    Ok(writer.0)
}

fn encode_payload<W: minicbor::encode::Write>(
    encoder: &mut Encoder<W>,
    ordinal: u32,
    first_logical_entity_ordinal: u64,
    media_type: MeshingChunkMediaType,
    schema_version: u16,
    fragments: super::packing::Fragments<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError>
where
    W::Error: std::fmt::Display,
{
    encoder
        .array(6)
        .and_then(|encoder| encoder.u16(CHUNK_ENVELOPE_SCHEMA_VERSION))
        .and_then(|encoder| encoder.str(media_type.media_type()))
        .and_then(|encoder| encoder.u16(schema_version))
        .and_then(|encoder| encoder.u32(ordinal))
        .and_then(|encoder| encoder.u64(first_logical_entity_ordinal))
        .and_then(|encoder| encoder.array(fragments.len() as u64))
        .map_err(encoding_error)?;
    for fragment in fragments.iter() {
        checkpoint(cancellation)?;
        encoder
            .array(2)
            .and_then(|encoder| encoder.bool(fragment.completes_record))
            .and_then(|encoder| encoder.bytes(fragment.bytes))
            .map_err(encoding_error)?;
    }
    Ok(())
}

fn checkpoint(cancellation: &dyn MeshingCancellationSignal) -> Result<(), MeshingContractError> {
    if cancellation.is_cancelled() {
        return Err(MeshingContractError::invalid(
            "meshing chunk processing",
            "operation was cancelled",
        ));
    }
    Ok(())
}

fn encoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("meshing chunk encoding", error.to_string())
}

fn decoding_error<E: std::fmt::Display>(error: E) -> MeshingContractError {
    MeshingContractError::invalid("meshing chunk decoding", error.to_string())
}
