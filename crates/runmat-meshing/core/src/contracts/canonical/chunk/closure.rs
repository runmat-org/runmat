mod layout;
mod prepared;
pub use prepared::{
    close_stage_payload_with_admission, ClosedMeshingStage, MeshingManifestMetadata,
};
mod reconstruction;
mod streaming;
use super::{
    cancellation_checkpoint, codec, validate_streams, EncodedMeshingChunk, MeshingChunkStream,
    MeshingChunkedPayload,
};
use crate::contracts::canonical::{
    CanonicalMeshingContract, MeshingCancellationSignal, MeshingContractError,
    MeshingManifestDisposition, MeshingStageKind, MeshingStageManifest, MeshingStageResultIdentity,
    MeshingStageResultKind, NeverCancelled, StableDigest, MESHING_IDENTITY_SCHEMA_VERSION,
    MESHING_STAGE_MANIFEST_SCHEMA_VERSION,
};
pub use reconstruction::{
    decode_stage_manifest_streams_borrowed_with_admission, MeshingStreamStorageProfile,
};
pub use streaming::{
    verify_stage_manifest_closure_borrowed, verify_stage_manifest_closure_borrowed_with_admission,
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
    prepared::prepare_manifest(
        MeshingManifestMetadata {
            stage,
            result_kind,
            producer_identity_digest,
            invariant_summary_digest,
            disposition,
        },
        prerequisite_manifest_digests,
        payload,
        &NeverCancelled,
        &mut |_| Ok(()),
    )
}

pub fn verify_stage_manifest_closure(
    manifest: &MeshingStageManifest,
    result_identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunk],
) -> Result<(), MeshingContractError> {
    let borrowed = chunks.iter().map(Into::into).collect::<Vec<_>>();
    verify_stage_manifest_closure_borrowed(manifest, result_identity, &borrowed, &NeverCancelled)
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
    let borrowed = chunks.iter().map(Into::into).collect::<Vec<_>>();
    decode_stage_manifest_streams_borrowed(manifest, result_identity, &borrowed, cancellation)
}

/// Decode admitted records while borrowing the encoded inventory.
pub fn decode_stage_manifest_streams_borrowed(
    manifest: &MeshingStageManifest,
    result_identity: &MeshingStageResultIdentity,
    chunks: &[super::EncodedMeshingChunkRef<'_>],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<MeshingChunkStream>, MeshingContractError> {
    decode_stage_manifest_streams_borrowed_with_admission(
        manifest,
        result_identity,
        chunks,
        cancellation,
        |_| Ok(()),
    )
}
