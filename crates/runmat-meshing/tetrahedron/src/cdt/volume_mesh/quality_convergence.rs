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
    DelaunayVolumeNode, DelaunayVolumeProvenance, DelaunayVolumeQuality,
    DelaunayVolumeSliverMutation, DelaunayVolumeTopology,
};
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;

mod repair;

pub(super) struct ConvergedVolumeQuality {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) provenance: DelaunayVolumeProvenance,
    pub(super) refinement_insertions: Vec<DelaunayVolumeNode>,
    pub(super) optimization_mutations: Vec<DelaunayVolumeSliverMutation>,
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
            refinement_insertions: Vec::new(),
            optimization_mutations: Vec::new(),
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
    let mut refinement_insertions = Vec::new();
    let mut optimization_mutations = Vec::new();
    let mut remaining_refinement = options.refinement.maximum_insertions;
    let mut remaining_optimization = options.sliver.maximum_passes;
    let repair = repair::RepairExecution {
        context,
        prepared: &prepared,
        options,
        cancellation,
        observer,
    };
    let mut blocked_repair = None;
    loop {
        let before = (refinement_insertions.len(), optimization_mutations.len());
        let mut refinement_stalled = false;
        let mut refinement_yielded = false;
        if state
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
                    refinement_insertions.extend(inserted);
                    // Rejection described the preceding physical state.
                    blocked_repair = None;
                }
                WorkingRefinementProgress::Yielded => refinement_yielded = true,
                WorkingRefinementProgress::Blocked => refinement_stalled = true,
            }
        }
        if refinement_stalled || state.quality().worst(QualityObjective::Shape).is_some() {
            if blocked_repair.is_none() {
                let repaired = repair.attempt(state, remaining_optimization)?;
                remaining_optimization =
                    remaining_optimization.saturating_sub(repaired.mutations.len() as u64);
                if !repaired.mutations.is_empty() {
                    search.repaired();
                }
                optimization_mutations.extend(repaired.mutations);
                state = repaired.state;
                blocked_repair = repaired.blocked;
            }
            // A local handoff is not global exhaustion. Give the remaining
            // unchanged refinement frontier its opportunity before rejecting.
            if refinement_stalled {
                if let Some(failure) = blocked_repair.take() {
                    return Err(repair.reject(failure, Some(Box::new(state))));
                }
            }
        }
        if state
            .quality()
            .worst(QualityObjective::Refinement)
            .is_none()
            && state.quality().worst(QualityObjective::Shape).is_none()
        {
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
                refinement_insertions,
                optimization_mutations,
                #[cfg(test)]
                lifecycle_counts,
            });
        }
        if refinement_yielded {
            continue;
        }
        if before == (refinement_insertions.len(), optimization_mutations.len()) {
            if let Some(failure) = blocked_repair.take() {
                return Err(repair.reject(failure, Some(Box::new(state))));
            }
            return Err(error(
                DelaunayVolumeMeshStage::Optimization,
                DelaunayVolumeMeshErrorKind::InvalidQuality,
                "coupled volume refinement and optimization made no progress",
            ));
        }
    }
}

#[cfg(test)]
mod tests;
