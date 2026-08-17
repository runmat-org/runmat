use super::{
    codec, logical_content_digest, validate_streams, EncodedMeshingChunk, MeshingChunkStream,
    MeshingChunkedPayload,
};
use crate::contracts::canonical::{
    CanonicalMeshingContract, MeshingContractError, MeshingManifestDisposition, MeshingStageKind,
    MeshingStageManifest, MeshingStageResultIdentity, MeshingStageResultKind, StableDigest,
    MESHING_IDENTITY_SCHEMA_VERSION, MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
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
    for (expected, chunk) in manifest.chunks.iter().zip(chunks) {
        if expected != &chunk.descriptor {
            return Err(MeshingContractError::invalid(
                "meshing stage manifest closure",
                "provided chunk inventory differs from the manifest",
            ));
        }
        let decoded = codec::decode_chunk(chunk)?;
        logical_entity_count = logical_entity_count
            .checked_add(decoded.records.len() as u64)
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
        match streams.last_mut() {
            Some(stream)
                if stream.media_type == decoded.media_type
                    && stream.schema_version == decoded.schema_version =>
            {
                stream.records.extend(decoded.records);
            }
            _ => streams.push(MeshingChunkStream {
                media_type: decoded.media_type,
                schema_version: decoded.schema_version,
                records: decoded.records,
            }),
        }
    }
    validate_streams(&streams)?;
    if logical_entity_count != result_identity.logical_entity_count
        || total_encoded_length != manifest.total_encoded_length
        || logical_content_digest(&streams)? != result_identity.logical_content_digest
    {
        return Err(MeshingContractError::invalid(
            "meshing stage manifest closure",
            "decoded logical content or encoded totals do not match the result identity",
        ));
    }
    Ok(())
}
