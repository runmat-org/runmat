//! Original typed mesh buffers and new preparation owners; codec work is a separate phase.
use super::StructuralStudyResult;
use crate::analysis::storage::AnalysisPublicationError;
pub(super) fn inventory(
    result: &StructuralStudyResult,
    maximum: u64,
) -> Result<(u64, u64), AnalysisPublicationError> {
    let mesh =
        crate::analysis::storage::publication::retained_mesh_storage(result.mesh(), maximum)?;
    let original = crate::analysis::storage::publication::typed_resident(
        &(
            result.meshing_evidence(),
            result.workloads(),
            result.numerical(),
        ),
        maximum,
    )?;
    // Intent resolution and workload dispatch introduce additional small metadata
    // owners. Compiled element/audit inventories share their validated originals.
    // Keep conservative existing profiles where no typed-buffer inventory exists.
    let copies = crate::analysis::storage::publication::typed_resident(
        &(
            &result.compiled().model,
            &result.compiled().model,
            &result.compiled().model,
            result.workloads(),
            result.workloads(),
            result.numerical(),
            result.meshing_evidence(),
        ),
        maximum,
    )?;
    let limits = runmat_canonical_codec::CanonicalLimits::content_bounded(
        usize::try_from(maximum).map_err(|_| AnalysisPublicationError::ResourceLimit {
            message: "typed storage budget exceeds address space".into(),
        })?,
        65_536,
        64,
    );
    let compiled_original = result
        .compiled()
        .retained_storage_bytes_estimate(limits)
        .map_err(|message| AnalysisPublicationError::ResourceLimit { message })?;
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "typed publication storage overflow".into(),
            })
    };
    Ok((
        add(add(original, compiled_original)?, mesh)?,
        add(copies, mesh)?,
    ))
}
