//! Size and shape scheduling share one accepted mutation/quality owner. Complete
//! canonical conversion occurs at terminal admission or a necessary global strategy.
use super::error::{quality_error, refinement_error, sliver_error};
use super::{
    error, DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshOptions,
    DelaunayVolumeMeshStage,
};
use crate::cdt::{
    volume_quality::{working::PreparedLocalQuality, QualityObjective},
    volume_refinement::{
        context::RefinementContext,
        working::{WorkingRefinement, WorkingRefinementProgress},
    },
    volume_state::{SearchState, WorkingState},
    DelaunayVolumeProvenance, DelaunayVolumeQuality, DelaunayVolumeSliverMutation,
    DelaunayVolumeTopology,
};
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;

mod repair;
mod schedule;
use schedule::{Next, RefinementStep, RepairStep};

pub(super) struct ConvergedVolumeQuality {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) provenance: DelaunayVolumeProvenance,
    pub(super) quality_mutations: Vec<super::DelaunayVolumeQualityMutation>,
    #[cfg(test)]
    pub(super) lifecycle_counts: (u64, u64),
}

pub(super) struct QualityExecution<'a> {
    pub(super) cancellation: &'a dyn MeshingCancellationSignal,
    pub(super) observer: &'a dyn super::quality_replay::VolumeQualityObserver,
}
#[cfg(test)]
impl<'a> QualityExecution<'a> {
    pub(super) fn without_observer(cancellation: &'a dyn MeshingCancellationSignal) -> Self {
        Self {
            cancellation,
            observer: &super::quality_replay::NoopQualityObserver,
        }
    }
}

pub(super) fn converge_volume_quality(
    topology: DelaunayVolumeTopology,
    quality: DelaunayVolumeQuality,
    metric_request: &MetricFieldRequest,
    provenance: DelaunayVolumeProvenance,
    fixed_node_identities: &[runmat_meshing_core::StableDigest],
    options: DelaunayVolumeMeshOptions,
    execution: QualityExecution<'_>,
) -> Result<ConvergedVolumeQuality, DelaunayVolumeMeshError> {
    let QualityExecution {
        cancellation,
        observer,
    } = execution;
    if quality.worst_refinement_tetrahedron.is_none()
        && quality.worst_optimization_tetrahedron.is_none()
    {
        return Ok(ConvergedVolumeQuality {
            topology,
            quality,
            provenance,
            quality_mutations: Vec::new(),
            #[cfg(test)]
            lifecycle_counts: (0, 0),
        });
    }
    let prepared = PreparedLocalQuality::prepare(metric_request, &provenance, options.quality)
        .map_err(quality_error)?;
    if let Some(conflict) = crate::cdt::volume_quality::boundary_compatibility::certify(
        &topology,
        &prepared,
        cancellation,
    )
    .map_err(quality_error)?
    {
        return Err(error(DelaunayVolumeMeshStage::Quality, DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint,
            format!("fixed PLC edge {:?} has a certified metric scaled-Jacobian upper bound {} below requested {}; surface remeshing is required", conflict.edge, conflict.jacobian_upper_bound, options.quality.minimum_metric_scaled_jacobian)));
    }
    let context = RefinementContext {
        metric_request,
        provenance: &provenance,
        quality_options: options.quality,
        fixed_node_identities,
    };
    let mut state = WorkingState::new(
        topology,
        quality,
        options.quality,
        &prepared,
        fixed_node_identities,
        cancellation,
    )
    .map_err(|failure| sliver_error(failure.into()))?;
    let mut search = WorkingRefinement::coupled();
    let mut quality_mutations = Vec::new();
    let mut remaining_refinement = options.refinement.maximum_insertions;
    let mut remaining_optimization = options.sliver.maximum_passes;
    let repair = repair::RepairExecution {
        context,
        prepared: &prepared,
        options,
        cancellation,
        observer,
    };
    let mut schedule = schedule::CoupledSchedule::default();
    // Repairs rerun after every accepted refinement; searches they already rejected rerun only
    // when their neighborhood changed.
    let mut rejected = crate::cdt::volume_sliver::RejectedCompleteRelocations::default();
    loop {
        let refinement = if state
            .quality()
            .worst(QualityObjective::Refinement)
            .is_some()
        {
            match search
                .step(
                    &mut state,
                    options.refinement.step,
                    remaining_refinement,
                    cancellation,
                )
                .map_err(refinement_error)?
            {
                WorkingRefinementProgress::Accepted(inserted) => {
                    remaining_refinement =
                        remaining_refinement.saturating_sub(inserted.len() as u64);
                    quality_mutations.extend(
                        inserted
                            .into_iter()
                            .map(super::DelaunayVolumeQualityMutation::Refined),
                    );
                    RefinementStep::Accepted
                }
                WorkingRefinementProgress::Yielded => RefinementStep::Yielded,
                WorkingRefinementProgress::Blocked => RefinementStep::Blocked,
            }
        } else {
            RefinementStep::Idle
        };
        schedule.refined(refinement);
        let shape_violation = state.quality().worst(QualityObjective::Shape).is_some();
        let mut repair_step = None;
        if schedule.should_repair(refinement, shape_violation) {
            let repaired = repair.attempt(state, remaining_optimization, &mut rejected)?;
            remaining_optimization =
                remaining_optimization.saturating_sub(repaired.mutations.len() as u64);
            let step = if repaired.mutations.is_empty() {
                RepairStep::Unchanged
            } else {
                search.repaired();
                RepairStep::Changed
            };
            quality_mutations.extend(
                repaired
                    .mutations
                    .into_iter()
                    .map(super::DelaunayVolumeQualityMutation::Optimized),
            );
            state = repaired.state;
            schedule.repaired(step, repaired.blocked);
            repair_step = Some(step);
        }
        let converged = state
            .quality()
            .worst(QualityObjective::Refinement)
            .is_none()
            && state.quality().worst(QualityObjective::Shape).is_none();
        match schedule.next(refinement, repair_step, converged) {
            Next::Continue => {}
            Next::Finish => {
                let (topology, quality) = state
                    .finish(options.sliver, cancellation)
                    .map_err(|failure| sliver_error(failure.into()))?;
                #[cfg(test)]
                let lifecycle_counts = prepared.lifecycle_counts();
                drop(prepared);
                return Ok(ConvergedVolumeQuality {
                    topology,
                    quality,
                    provenance,
                    quality_mutations,
                    #[cfg(test)]
                    lifecycle_counts,
                });
            }
            Next::Reject(failure) => return Err(repair.reject(failure, Some(Box::new(state)))),
            Next::NoProgress => {
                return Err(error(
                    DelaunayVolumeMeshStage::Optimization,
                    DelaunayVolumeMeshErrorKind::InvalidQuality,
                    "coupled volume refinement and optimization made no progress",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests;
