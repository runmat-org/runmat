//! Non-chunk publication storage is distinct from the already admitted payload.
use super::*;
use runmat_execution_artifact::{ArtifactAccess, ArtifactError, ObjectDescriptor};
pub(super) fn slots<T>(count: usize) -> crate::MeshingExecutionResult<u64> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| ArtifactError::Limit("publication metadata storage overflow".into()).into())
}
pub(super) fn admit_evidence(
    evidence: &runmat_meshing_core::MeshingStageEvidence,
    access: &ArtifactAccess,
    admit: &mut impl FnMut(u64) -> crate::MeshingExecutionResult<()>,
) -> crate::MeshingExecutionResult<()> {
    admit(
        evidence
            .canonical_encoding_profile()?
            .resident_workspace_bytes_estimate,
    )?;
    admit(ObjectDescriptor::content_named_text_bytes(
        "meshing/stage-evidence",
        crate::stage_evidence_object::STAGE_EVIDENCE_MEDIA_TYPE,
    )?)?;
    admit(access.reference_text_bytes(
        crate::stage_evidence_object::STAGE_EVIDENCE_MEDIA_TYPE,
        crate::stage_evidence_object::STAGE_EVIDENCE_VALUE_SCHEMA,
    )?)?;
    Ok(())
}
