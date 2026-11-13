//! Physical admission precedes completeness repair of a locally inserted seed.
//! Valid constrained connectivity needs no canonical tie-breaking rearrangement.
use super::*;
use crate::cdt::insertion::legalize_delaunay_volume_topology_completely_with_barriers;

pub(super) fn admit_or_repair(
    candidate: DelaunayVolumeTopology,
    protected_faces: &[[StableDigest; 3]],
    options: DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayInsertionError> {
    match validate_constrained_delaunay_volume_topology(
        &candidate,
        protected_faces,
        options,
        cancellation,
    ) {
        Ok(()) => return Ok(candidate),
        Err(failure) if seed_is_unavailable(&failure) => {}
        Err(failure) => return Err(failure),
    }
    // Keep the complete fallback for a physical violation. Neither insertion
    // nor repair success can replace independent terminal admission.
    let repaired = legalize_delaunay_volume_topology_completely_with_barriers(
        candidate,
        protected_faces,
        options,
        cancellation,
    )?;
    validate_constrained_delaunay_volume_topology(
        &repaired,
        protected_faces,
        options,
        cancellation,
    )?;
    Ok(repaired)
}

#[cfg(test)]
mod tests;
