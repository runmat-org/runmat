//! Admit a reconnected edge only after restoring and checking its surrounding topology.
use super::{
    diagnostic, edge_exists, error, DelaunayInsertionError, DelaunayInsertionErrorKind,
    DelaunaySegmentRecoveryError, DelaunaySegmentRecoveryErrorKind, DelaunayVolumeTopology,
    RecoveryWork, StableDigest,
};
use crate::cdt::insertion::restore_constrained_delaunay_topology;

pub(super) fn admit(
    candidate: DelaunayVolumeTopology,
    endpoints: [StableDigest; 2],
    protected_edges: &[[StableDigest; 2]],
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    let Some(candidate) = restore(candidate, protected_faces, constraint_index, work)? else {
        return Ok(None);
    };
    Ok(preserves_edges(
        &candidate,
        endpoints,
        protected_edges,
        constraint_index,
        work,
    )?
    .then_some(candidate))
}

pub(super) fn preserves_edges(
    candidate: &impl crate::cdt::insertion::topology_access::TetrahedralTopology,
    endpoints: [StableDigest; 2],
    protected_edges: &[[StableDigest; 2]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<bool, DelaunaySegmentRecoveryError> {
    // Independently check edge preservation rather than deriving it from the
    // current legalization implementation's choice of flips.
    for [left, right] in std::iter::once(&endpoints).chain(protected_edges) {
        if !edge_exists(candidate, *left, *right, constraint_index, work)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Restore shared empty-sphere admission after reconnection or a batch of insertions.
/// Callers independently check their complete required-edge inventory afterward.
pub(super) fn restore(
    candidate: DelaunayVolumeTopology,
    protected_faces: &[[StableDigest; 3]],
    constraint_index: u32,
    work: &mut RecoveryWork<'_>,
) -> Result<Option<DelaunayVolumeTopology>, DelaunaySegmentRecoveryError> {
    match restore_constrained_delaunay_topology(
        candidate,
        protected_faces,
        work.options.insertion,
        work.cancellation,
    ) {
        Ok(candidate) => Ok(Some(candidate)),
        Err(failure) => {
            reject_or_propagate(&failure, constraint_index)?;
            Ok(None)
        }
    }
}

pub(super) fn reject_or_propagate(
    failure: &DelaunayInsertionError,
    constraint_index: u32,
) -> Result<(), DelaunaySegmentRecoveryError> {
    let kind = match failure.kind {
        DelaunayInsertionErrorKind::ResourceLimit => {
            DelaunaySegmentRecoveryErrorKind::ResourceLimit
        }
        DelaunayInsertionErrorKind::Cancelled => DelaunaySegmentRecoveryErrorKind::Cancelled,
        _ => {
            diagnostic::trace_rejected_candidate(constraint_index, failure);
            return Ok(());
        }
    };
    Err(error(kind, Some(constraint_index), failure.to_string()))
}
