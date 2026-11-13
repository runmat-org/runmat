//! Typed requested heap inventories for one target, excluding borrowed original topology.
use super::*;
use crate::topology::storage::{add, slots, DenseIndexUnion, TopologyStorageError};

pub(in crate::results::observation::resolve::targets) fn storage_error(
    error: TopologyStorageError,
) -> ObservationError {
    ObservationError::Topology(error.into())
}
pub(in crate::results::observation::resolve::targets) fn require_workspace(
    bytes: u64,
    limits: ObservationLimits,
) -> Result<(), ObservationError> {
    if bytes > limits.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit(
            "target selection construction residency",
        ));
    }
    Ok(())
}
pub(in crate::results::observation::resolve::targets) fn selection_output_bytes(
    count: usize,
    maximum: usize,
) -> Result<u64, ObservationError> {
    add(
        slots::<usize>(count.min(maximum)).map_err(storage_error)?,
        slots::<StableDigest>(count.min(maximum)).map_err(storage_error)?,
    )
    .map_err(storage_error)
}
pub(in crate::results::observation::resolve::targets) fn node_union_resident_bytes(
    domain: usize,
    maximum: usize,
) -> Result<u64, ObservationError> {
    // Bitmap and index output coexist; identity output is constructed after bitmap release.
    Ok(DenseIndexUnion::resident_bytes(domain, maximum)
        .map_err(storage_error)?
        .max(selection_output_bytes(domain, maximum)?))
}
pub(in crate::results::observation::resolve::targets) fn element_selection_resident_bytes(
    target: &runmat_analysis_core::RegionTarget,
    topology: &ObservationTopology,
    projection: Option<&AnalysisProjection>,
    target_bytes: u64,
) -> Result<u64, ObservationError> {
    let coverage = slots::<bool>(target.resolved_entities().len()).map_err(storage_error)?;
    let mut source = 0;
    if let (Some(layout), Some(projection)) = (&topology.finite_volume, projection) {
        // Sorted source identity map + retained canonical cell-to-source indices + Arc header.
        source = add(
            slots::<(StableDigest, usize)>(projection.elements.len()).map_err(storage_error)?,
            slots::<usize>(layout.cell_identities().len()).map_err(storage_error)?,
        )
        .map_err(storage_error)?;
        source = add(
            source,
            (std::mem::size_of::<Vec<usize>>() + 2 * std::mem::size_of::<usize>()) as u64,
        )
        .map_err(storage_error)?;
    }
    // Coverage is dropped before the source map; summing these bounds is conservative.
    add(add(coverage, source).map_err(storage_error)?, target_bytes).map_err(storage_error)
}
pub(in crate::results::observation::resolve::targets) fn face_selection_resident_bytes(
    target: &runmat_analysis_core::RegionTarget,
    projection: &AnalysisProjection,
    topology: &ObservationTopology,
    limits: ObservationLimits,
) -> Result<u64, ObservationError> {
    let faces = matching_faces(target, projection).count();
    let keys = slots::<usize>(faces).map_err(storage_error)?;
    let coverage = slots::<bool>(target.resolved_entities().len()).map_err(storage_error)?;
    let output = selection_output_bytes(
        topology.face_boundary_sources.len(),
        limits.maximum_selected_entities,
    )?;
    add(add(keys, coverage).map_err(storage_error)?, output).map_err(storage_error)
}

#[cfg(test)]
mod tests;
