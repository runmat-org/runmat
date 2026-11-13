//! One immutable layout validates each chunk once. Logical record boundaries are
//! retained as lengths, never as an unbounded list of borrowed fragment pointers.
use super::*;
use crate::EncodedMeshingChunkRef;
use sha2::{Digest as _, Sha256};

pub(super) struct RecordLayout<'a> {
    pub chunks: Vec<codec::DecodedMeshingChunk<'a>>,
    pub lengths: Vec<usize>,
    pub streams: usize,
}

pub(super) fn storage_profile(
    manifest: &MeshingStageManifest,
    identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunkRef<'_>],
    materialize: bool,
) -> Result<MeshingStreamStorageProfile, MeshingContractError> {
    if chunks.is_empty()
        || chunks.len() > super::super::MAX_CHUNKS
        || manifest.chunks.len() != chunks.len()
    {
        return Err(invalid(
            "provided chunk inventory differs from the bounded manifest",
        ));
    }
    let mut records = 0;
    let mut payload = 0;
    let mut encoded = 0;
    let mut streams = 0;
    let mut previous = None;
    for (expected, chunk) in manifest.chunks.iter().zip(chunks) {
        if expected != chunk.descriptor {
            return Err(invalid(
                "provided chunk inventory differs from the manifest",
            ));
        }
        let key = (expected.media_type, expected.schema_version);
        if previous != Some(key) {
            streams = add(streams, 1)?;
            previous = Some(key);
        }
        records = add(records, expected.logical_entity_count)?;
        payload = add(payload, expected.decoded_length)?;
        encoded = add(encoded, expected.encoded_length)?;
    }
    if records != identity.logical_entity_count || encoded != manifest.total_encoded_length {
        return Err(invalid(
            "manifest record or encoded totals differ from the result identity",
        ));
    }
    let identity_encoding = identity.canonical_encoding_profile()?;
    Ok(MeshingStreamStorageProfile {
        retained_bytes: if materialize {
            add(
                payload,
                add(
                    slots::<Vec<u8>>(records)?,
                    slots::<MeshingChunkStream>(streams)?,
                )?,
            )?
        } else {
            0
        },
        scratch_bytes: add(
            slots::<usize>(records)?,
            add(
                slots::<codec::DecodedMeshingChunk<'_>>(chunks.len() as u64)?,
                add(
                    MeshingStageManifest::validation_scratch_bytes(chunks.len())?,
                    identity_encoding.resident_workspace_bytes_estimate,
                )?,
            )?,
        )?,
    })
}

pub(super) fn prepare<'a>(
    manifest: &MeshingStageManifest,
    identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunkRef<'a>],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<RecordLayout<'a>, MeshingContractError> {
    cancellation_checkpoint(cancellation)?;
    manifest.validate()?;
    identity.validate()?;
    if manifest.stage != identity.stage
        || manifest.result_kind != identity.result_kind
        || manifest.invariant_summary_digest != identity.invariant_summary_digest
        || manifest.logical_result_identity != identity.canonical_digest()?
    {
        return Err(invalid(
            "manifest metadata does not close over the logical stage result",
        ));
    }
    let count = usize::try_from(identity.logical_entity_count)
        .map_err(|_| invalid("record count does not fit in memory"))?;
    // A raw descriptor's logical count is only a claim. Verify bounded original
    // chunk grammar/counts before using that count for a record-length array.
    let mut decoded_chunks = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        decoded_chunks.push(codec::decode_chunk_with_cancellation(*chunk, cancellation)?);
    }
    let mut lengths = Vec::with_capacity(count);
    let mut pending = 0usize;
    let mut record_open = false;
    let mut previous = None;
    let mut stream_records = 0usize;
    let mut streams = 0usize;
    for decoded in &decoded_chunks {
        let key = (decoded.media_type, decoded.schema_version);
        if previous != Some(key) {
            if record_open
                || previous.is_some_and(|(media, _)| media >= key.0)
                || (previous.is_some() && stream_records == 0)
                || key.1 == 0
            {
                return Err(invalid(
                    "fragmented record crosses a stream boundary or streams are empty/unordered",
                ));
            }
            previous = Some(key);
            stream_records = 0;
            streams += 1;
        }
        for fragment in decoded.fragments.iter() {
            cancellation_checkpoint(cancellation)?;
            record_open = true;
            pending = pending
                .checked_add(fragment.bytes.len())
                .ok_or_else(|| invalid("record length overflow"))?;
            if fragment.completes_record {
                if pending == 0 || lengths.len() == count {
                    return Err(invalid(
                        "completed record is empty or exceeds its inventory",
                    ));
                }
                lengths.push(pending);
                pending = 0;
                record_open = false;
                stream_records += 1;
            }
        }
    }
    if record_open || lengths.len() != count || stream_records == 0 {
        return Err(invalid(
            "final fragment is incomplete or record inventory differs",
        ));
    }
    let result = RecordLayout {
        chunks: decoded_chunks,
        lengths,
        streams,
    };
    verify_content(&result, identity, cancellation)?;
    Ok(result)
}
fn verify_content(
    layout: &RecordLayout<'_>,
    identity: &MeshingStageResultIdentity,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    let mut hasher = Sha256::new();
    hasher.update(b"runmat-meshing-logical-content/v1\0");
    hasher.update(identity.logical_entity_count.to_be_bytes());
    let mut ordinal = 0usize;
    let mut record_open = false;
    for decoded in &layout.chunks {
        for fragment in decoded.fragments.iter() {
            cancellation_checkpoint(cancellation)?;
            if !record_open {
                let media = decoded.media_type.media_type().as_bytes();
                hasher.update((ordinal as u64).to_be_bytes());
                hasher.update((media.len() as u32).to_be_bytes());
                hasher.update(media);
                hasher.update(decoded.schema_version.to_be_bytes());
                hasher.update((layout.lengths[ordinal] as u64).to_be_bytes());
                record_open = true;
            }
            for block in fragment.bytes.chunks(1024 * 1024) {
                cancellation_checkpoint(cancellation)?;
                hasher.update(block);
            }
            if fragment.completes_record {
                ordinal += 1;
                record_open = false;
            }
        }
    }
    if StableDigest::from_bytes(hasher.finalize().into()) != identity.logical_content_digest {
        return Err(invalid(
            "decoded logical content does not match the result identity",
        ));
    }
    Ok(())
}
fn invalid(reason: &'static str) -> MeshingContractError {
    MeshingContractError::invalid("meshing stage manifest closure", reason)
}
fn add(a: u64, b: u64) -> Result<u64, MeshingContractError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("closure storage overflow"))
}
fn slots<T>(count: u64) -> Result<u64, MeshingContractError> {
    count
        .checked_mul(std::mem::size_of::<T>() as u64)
        .ok_or_else(|| invalid("closure storage overflow"))
}
