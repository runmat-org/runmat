use super::{
    cancellation_checkpoint, codec, logical_content_digest_with_cancellation, validate_streams,
    EncodedMeshingChunk, MeshingChunkStream, MeshingChunkedPayload,
};
use crate::contracts::canonical::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingContractError,
    MeshingManifestDisposition, MeshingStageKind, MeshingStageManifest, MeshingStageResultIdentity,
    MeshingStageResultKind, NeverCancelled, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
};

pub fn build_closed_stage_manifest(
    stage: MeshingStageKind,
    result_kind: MeshingStageResultKind,
    producer_identity_digest: StableDigest,
    invariant_summary_digest: StableDigest,
    prerequisite_manifest_digests: Vec<StableDigest>,
    disposition: MeshingManifestDisposition,
    payload: &MeshingChunkedPayload,
) -> Result<(MeshingStageResultIdentity, MeshingStageManifest), MeshingContractError> {
    let result_identity = MeshingStageResultIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage,
        result_kind,
        producer_identity_digest,
        logical_content_digest: payload.logical_content_digest,
        logical_entity_count: payload.logical_entity_count,
        invariant_summary_digest,
    };
    result_identity.validate()?;
    let manifest = MeshingStageManifest {
        schema_version: MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        stage,
        result_kind,
        logical_result_identity: result_identity.canonical_digest()?,
        disposition,
        prerequisite_manifest_digests,
        invariant_summary_digest,
        chunks: payload
            .chunks
            .iter()
            .map(|chunk| chunk.descriptor.clone())
            .collect(),
        total_encoded_length: payload.total_encoded_length,
    };
    manifest.validate()?;
    verify_stage_manifest_closure(&manifest, &result_identity, &payload.chunks)?;
    Ok((result_identity, manifest))
}

pub fn verify_stage_manifest_closure(
    manifest: &MeshingStageManifest,
    result_identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunk],
) -> Result<(), MeshingContractError> {
    decode_stage_manifest_streams(manifest, result_identity, chunks).map(|_| ())
}

/// Reconstructs typed logical streams only after independently verifying the complete closure.
pub fn decode_stage_manifest_streams(
    manifest: &MeshingStageManifest,
    result_identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunk],
) -> Result<Vec<MeshingChunkStream>, MeshingContractError> {
    decode_stage_manifest_streams_with_cancellation(
        manifest,
        result_identity,
        chunks,
        &NeverCancelled,
    )
}

/// Reconstructs and verifies typed logical streams while observing an execution-owned
/// cancellation signal during potentially large chunk and record operations.
pub fn decode_stage_manifest_streams_with_cancellation(
    manifest: &MeshingStageManifest,
    result_identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunk],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MeshingChunkStream>, MeshingContractError> {
    cancellation_checkpoint(cancellation)?;
    manifest.validate()?;
    result_identity.validate()?;
    if manifest.stage != result_identity.stage
        || manifest.result_kind != result_identity.result_kind
        || manifest.invariant_summary_digest != result_identity.invariant_summary_digest
        || manifest.logical_result_identity != result_identity.canonical_digest()?
        || manifest.chunks.len() != chunks.len()
    {
        return Err(MeshingContractError::invalid(
            "meshing stage manifest closure",
            "manifest metadata does not close over the logical stage result",
        ));
    }

    let mut streams = Vec::<MeshingChunkStream>::new();
    let mut logical_entity_count = 0_u64;
    let mut total_encoded_length = 0_u64;
    let mut pending_record = Vec::new();
    for (expected, chunk) in manifest.chunks.iter().zip(chunks) {
        cancellation_checkpoint(cancellation)?;
        if expected != &chunk.descriptor {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest closure",
                "provided chunk inventory differs from the manifest",
            ));
        }
        let decoded = codec::decode_chunk_with_cancellation(chunk, cancellation)?;
        logical_entity_count = logical_entity_count
            .checked_add(chunk.descriptor.logical_entity_count)
            .ok_or_else(|| {
                MeshingContractError::invalid(
                    "meshing stage manifest closure",
                    "logical entity count overflow",
                )
            })?;
        total_encoded_length = total_encoded_length
            .checked_add(chunk.descriptor.encoded_length)
            .ok_or_else(|| {
                MeshingContractError::invalid(
                    "meshing stage manifest closure",
                    "encoded byte count overflow",
                )
            })?;
        let stream = match streams.last_mut() {
            Some(stream)
                if stream.media_type == decoded.media_type
                    && stream.schema_version == decoded.schema_version =>
            {
                stream
            }
            _ if pending_record.is_empty() => {
                streams.push(MeshingChunkStream {
                    media_type: decoded.media_type,
                    schema_version: decoded.schema_version,
                    records: Vec::new(),
                });
                streams.last_mut().expect("stream was appended")
            }
            _ => {
                return Err(MeshingContractError::invalid(
                    "meshing stage manifest closure",
                    "a fragmented logical record crosses a stream boundary",
                ));
            }
        };
        let mut completed_in_chunk = 0_u64;
        for fragment in decoded.fragments {
            for block in fragment.bytes.chunks(1024 * 1024) {
                cancellation_checkpoint(cancellation)?;
                pending_record.extend_from_slice(block);
            }
            if fragment.completes_record {
                if pending_record.is_empty() {
                    return Err(MeshingContractError::invalid(
                        "meshing stage manifest closure",
                        "a completed logical record is empty",
                    ));
                }
                stream.records.push(std::mem::take(&mut pending_record));
                completed_in_chunk += 1;
            }
        }
        if completed_in_chunk != chunk.descriptor.logical_entity_count {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest closure",
                "record fragment completion count differs from its descriptor",
            ));
        }
    }
    cancellation_checkpoint(cancellation)?;
    if !pending_record.is_empty() {
        return Err(MeshingContractError::invalid(
            "meshing stage manifest closure",
            "final logical record fragment is incomplete",
        ));
    }
    validate_streams(&streams)?;
    if logical_entity_count != result_identity.logical_entity_count
        || total_encoded_length != manifest.total_encoded_length
        || logical_content_digest_with_cancellation(&streams, cancellation)?
            != result_identity.logical_content_digest
    {
        return Err(MeshingContractError::invalid(
            "meshing stage manifest closure",
            "decoded logical content or encoded totals do not match the result identity",
        ));
    }
    Ok(streams)
}
