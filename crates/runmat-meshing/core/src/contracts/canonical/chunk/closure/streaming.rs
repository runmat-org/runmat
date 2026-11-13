//! Stream original content through one admitted, immutable validated layout.
use super::*;
use crate::EncodedMeshingChunkRef;
pub fn verify_stage_manifest_closure_borrowed(
    manifest: &MeshingStageManifest,
    identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunkRef<'_>],
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), MeshingContractError> {
    verify_stage_manifest_closure_borrowed_with_admission(
        manifest,
        identity,
        chunks,
        cancellation,
        |_| Ok(()),
    )
}
/// Admit all layout, uniqueness and identity-codec scratch before allocation.
/// Scratch dies before return; the callback belongs to the caller's resource scope.
pub fn verify_stage_manifest_closure_borrowed_with_admission(
    manifest: &MeshingStageManifest,
    identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunkRef<'_>],
    cancellation: &dyn MeshingCancellationSignal,
    mut admit: impl FnMut(MeshingStreamStorageProfile) -> Result<(), MeshingContractError>,
) -> Result<(), MeshingContractError> {
    cancellation_checkpoint(cancellation)?;
    admit(layout::storage_profile(manifest, identity, chunks, false)?)?;
    layout::prepare(manifest, identity, chunks, cancellation)?;
    Ok(())
}
