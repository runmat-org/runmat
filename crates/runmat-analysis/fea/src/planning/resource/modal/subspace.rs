//! Checked inventory of the governed modal search, including boundary probes.
use super::{count, ResourcePlanningError, StructuralModalWorkload};
use crate::solve::modal::MODAL_BOUNDARY_PROBE_WIDTH;

pub(super) struct ModalSubspaceBounds {
    pub requested_mode_count: u64,
    pub maximum_subspace_dimension: u64,
    pub initial_subspace_dimension: u64,
    pub boundary_probe_width: u64,
    pub expansion_count: u64,
}

pub(super) fn derive(
    reduced: usize,
    workload: &StructuralModalWorkload,
) -> Result<ModalSubspaceBounds, ResourcePlanningError> {
    let requested_mode_count = count(
        workload.truncation().requested_mode_count().min(reduced),
        "requested modal count",
    )?;
    let maximum_subspace_dimension = count(
        workload
            .convergence()
            .admitted_subspace_dimension(reduced, workload.truncation().requested_mode_count())
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        "modal admitted subspace dimension",
    )?;
    let boundary_probe_width = count(MODAL_BOUNDARY_PROBE_WIDTH, "modal boundary probe width")?;
    let initial_subspace_dimension = requested_mode_count
        .checked_add(boundary_probe_width)
        .ok_or(ResourcePlanningError::Overflow("modal initial subspace"))?
        .min(maximum_subspace_dimension);
    let remaining = maximum_subspace_dimension - initial_subspace_dimension;
    let expansion_count = remaining
        .checked_add(boundary_probe_width - 1)
        .ok_or(ResourcePlanningError::Overflow("modal subspace expansions"))?
        / boundary_probe_width
        + 1;
    Ok(ModalSubspaceBounds {
        requested_mode_count,
        maximum_subspace_dimension,
        initial_subspace_dimension,
        boundary_probe_width,
        expansion_count,
    })
}
