//! Public admission and owned entry points for the refinement state machine.
use super::*;

pub fn refine_delaunay_volume(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinement, DelaunayVolumeRefinementStepError> {
    let progress = refine_delaunay_volume_until_stalled(input, options, cancellation)?;
    if progress.stalled {
        return Err(error(
            DelaunayVolumeRefinementStepErrorKind::ResourceLimit,
            "volume refinement exhausted every admissible construction before convergence",
        ));
    }
    validate_delaunay_volume_refinement(input, &progress.refinement, options, cancellation)?;
    Ok(progress.refinement)
}

pub(in crate::cdt) struct DelaunayVolumeRefinementProgress {
    pub(in crate::cdt) refinement: DelaunayVolumeRefinement,
    pub(in crate::cdt) stalled: bool,
}

pub(in crate::cdt) fn refine_delaunay_volume_until_stalled(
    input: DelaunayVolumeRefinementInput<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinementProgress, DelaunayVolumeRefinementStepError> {
    refine_owned_until_stalled(
        input.topology.clone(),
        input.quality.clone(),
        input.into(),
        options,
        cancellation,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::cdt) enum RefinementSchedule {
    Exhaustive,
    Cooperative,
}

pub(in crate::cdt) fn refine_owned_until_stalled(
    topology: crate::cdt::DelaunayVolumeTopology,
    quality: crate::cdt::DelaunayVolumeQuality,
    context: super::super::context::RefinementContext<'_>,
    options: DelaunayVolumeRefinementOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeRefinementProgress, DelaunayVolumeRefinementStepError> {
    refine_owned_with_schedule(
        topology,
        quality,
        context,
        options,
        cancellation,
        RefinementSchedule::Exhaustive,
    )
}
