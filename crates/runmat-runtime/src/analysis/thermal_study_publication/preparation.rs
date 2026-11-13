//! Original thermal ownership and introduced completion copies, before replay.
use super::*;
use crate::analysis::{storage, study_publication::preparation::PublicationPreparation};
pub(super) fn capture(
    definition: Digest,
    completed: &ThermalStudyCompletion,
    mesh: &SolverMeshArtifact,
    evidence: &MeshingEvidence,
    options: &AnalysisThermalRunOptions,
) -> Result<
    (
        PublicationPreparation,
        storage::publication::OwnedCopyReservation,
    ),
    storage::AnalysisPublicationError,
> {
    let maximum = crate::analysis::configuration::publication_staging_budget()?;
    let mesh_bytes = storage::publication::retained_mesh_storage(mesh, maximum)?;
    let original_metadata = storage::publication::resident(&(evidence, options), maximum)?;
    let limits = runmat_canonical_codec::CanonicalLimits::content_bounded(
        usize::try_from(maximum).map_err(|_| storage::AnalysisPublicationError::ResourceLimit {
            message: "thermal preparation exceeds address space".into(),
        })?,
        65_536,
        64,
    );
    let completion_copy = completed
        .shared_clone_storage_bytes_estimate(limits)
        .map_err(|message| storage::AnalysisPublicationError::ResourceLimit { message })?;
    let metadata_copies = storage::publication::resident(
        &(
            &completed.compiled.model,
            &completed.compiled.model,
            &completed.workload,
            &completed.workload,
            evidence,
            options,
        ),
        maximum,
    )?;
    let original = completed
        .retained_storage_bytes_estimate(limits)
        .map_err(|message| storage::AnalysisPublicationError::ResourceLimit { message })?
        .checked_add(original_metadata)
        .and_then(|bytes| bytes.checked_add(mesh_bytes))
        .ok_or_else(|| storage::AnalysisPublicationError::ResourceLimit {
            message: "thermal original storage overflow".into(),
        })?;
    let copies = completion_copy
        .checked_add(metadata_copies)
        .ok_or_else(|| storage::AnalysisPublicationError::ResourceLimit {
            message: "thermal copy storage overflow".into(),
        })?;
    PublicationPreparation::capture_with_completion_storage(definition, original, copies, maximum)
}
