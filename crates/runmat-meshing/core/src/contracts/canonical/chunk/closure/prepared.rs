//! Construction-only closure authority. Raw/serialized parts cannot recreate it
//! without independently validating their bytes. It carries no restart authority.
use super::*;
use crate::EncodedMeshingChunkRef;

pub struct MeshingManifestMetadata {
    pub stage: MeshingStageKind,
    pub result_kind: MeshingStageResultKind,
    pub producer_identity_digest: StableDigest,
    pub invariant_summary_digest: StableDigest,
    pub disposition: MeshingManifestDisposition,
}

pub struct ClosedMeshingStage {
    identity: MeshingStageResultIdentity,
    manifest: MeshingStageManifest,
    chunks: Vec<EncodedMeshingChunk>,
}
impl ClosedMeshingStage {
    /// Consuming this authority returns raw parts; changing them loses the proof.
    pub fn into_parts(
        self,
    ) -> (
        MeshingStageResultIdentity,
        MeshingStageManifest,
        Vec<EncodedMeshingChunk>,
    ) {
        (self.identity, self.manifest, self.chunks)
    }
}

/// Calls admission before each new typed buffer or codec/layout workspace. The
/// caller retains the charge while the returned manifest/chunks are live, and
/// may conservatively hold the smaller scratch allowances through publication.
pub fn close_stage_payload_with_admission(
    metadata: MeshingManifestMetadata,
    prerequisites: Vec<StableDigest>,
    payload: MeshingChunkedPayload,
    cancellation: &dyn MeshingCancellationSignal,
    mut admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
) -> Result<ClosedMeshingStage, MeshingContractError> {
    let (identity, manifest) =
        prepare_manifest(metadata, prerequisites, &payload, cancellation, &mut admit)?;
    Ok(ClosedMeshingStage {
        identity,
        manifest,
        chunks: payload.chunks,
    })
}

pub(super) fn prepare_manifest(
    metadata: MeshingManifestMetadata,
    prerequisites: Vec<StableDigest>,
    payload: &MeshingChunkedPayload,
    cancellation: &dyn MeshingCancellationSignal,
    admit: &mut impl FnMut(u64) -> Result<(), MeshingContractError>,
) -> Result<(MeshingStageResultIdentity, MeshingStageManifest), MeshingContractError> {
    cancellation_checkpoint(cancellation)?;
    if payload.chunks.is_empty() || payload.chunks.len() > super::super::MAX_CHUNKS {
        return Err(invalid(
            "stage payload must have a bounded nonempty chunk inventory",
        ));
    }
    let identity = MeshingStageResultIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage: metadata.stage,
        result_kind: metadata.result_kind,
        producer_identity_digest: metadata.producer_identity_digest,
        logical_content_digest: payload.logical_content_digest,
        logical_entity_count: payload.logical_entity_count,
        invariant_summary_digest: metadata.invariant_summary_digest,
    };
    identity.validate()?;
    admit(
        identity
            .canonical_encoding_profile()?
            .resident_workspace_bytes_estimate,
    )?;
    let descriptors = payload
        .chunks
        .len()
        .checked_mul(std::mem::size_of::<crate::MeshingChunkDescriptor>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("manifest descriptor storage overflow"))?;
    admit(descriptors)?;
    let manifest = MeshingStageManifest {
        schema_version: MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
        stage: metadata.stage,
        result_kind: metadata.result_kind,
        logical_result_identity: identity.canonical_digest()?,
        disposition: metadata.disposition,
        prerequisite_manifest_digests: prerequisites,
        invariant_summary_digest: metadata.invariant_summary_digest,
        chunks: payload
            .chunks
            .iter()
            .map(|chunk| chunk.descriptor.clone())
            .collect(),
        total_encoded_length: payload.total_encoded_length,
    };
    let views = payload
        .chunks
        .len()
        .checked_mul(std::mem::size_of::<EncodedMeshingChunkRef<'_>>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| invalid("manifest chunk-view storage overflow"))?;
    admit(views)?;
    let chunks = payload.chunks.iter().map(Into::into).collect::<Vec<_>>();
    verify_stage_manifest_closure_borrowed_with_admission(
        &manifest,
        &identity,
        &chunks,
        cancellation,
        |profile| admit(profile.requested_bytes()?),
    )?;
    Ok((identity, manifest))
}
fn invalid(reason: &'static str) -> MeshingContractError {
    MeshingContractError::invalid("meshing stage manifest construction", reason)
}

#[cfg(test)]
mod tests;
