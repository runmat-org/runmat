//! Pending original wire metadata remains owned until each closure transfers to replay.
use super::*;

pub(super) fn admit_pending<T: serde::Serialize>(
    study: &mut NonlinearSolidStudyCoordinator,
    caller_bytes: u64,
    pending: &T,
) -> AnalysisBridgeResult<()> {
    let ceiling = study.preparation.product.workload.model.sequence.steps[0]
        .accuracy
        .resources
        .resident_byte_budget();
    let profile = runmat_canonical_codec::profile_serialized(
        pending,
        runmat_canonical_codec::CanonicalLimits::content_bounded(
            usize::try_from(ceiling)
                .map_err(|_| limit("nonlinear pending scope exceeds address space"))?,
            65_536,
            64,
        ),
    )?;
    study.external_retained_bytes = caller_bytes
        .checked_add(profile.materialization_resident_bytes_estimate)
        .ok_or_else(|| limit("nonlinear pending metadata overflow"))?;
    study.admit_retention(None, None)?;
    Ok(())
}
