//! Coupled repair retains an unchanged rejected state for the other strategy.
//! Numerical rejection becomes terminal only when refinement cannot unlock it.
use super::*;
use crate::cdt::volume_sliver::{
    run_in_state, DelaunayVolumeSliverError, DelaunayVolumeSliverErrorKind, QualityConvergenceMode,
    RejectedCompleteRelocations,
};

pub(super) struct RepairExecution<'a, 'ctx> {
    pub context: RefinementContext<'ctx>,
    pub prepared: &'ctx PreparedLocalQuality<'ctx>,
    pub options: DelaunayVolumeMeshOptions,
    pub cancellation: &'a dyn MeshingCancellationSignal,
    pub observer: &'a dyn super::super::quality_replay::VolumeQualityObserver,
}
pub(super) struct RepairOutcome<'ctx> {
    pub state: WorkingState<'ctx>,
    pub mutations: Vec<DelaunayVolumeSliverMutation>,
    pub blocked: Option<DelaunayVolumeSliverError>,
}
impl<'ctx> RepairExecution<'_, 'ctx> {
    pub fn attempt(
        &self,
        state: WorkingState<'ctx>,
        remaining: u64,
        rejected: &mut RejectedCompleteRelocations,
    ) -> Result<RepairOutcome<'ctx>, DelaunayVolumeMeshError> {
        if remaining == 0 {
            return Err(error(
                DelaunayVolumeMeshStage::Optimization,
                DelaunayVolumeMeshErrorKind::ResourceLimit,
                "coupled volume quality convergence exhausted its optimization pass budget",
            ));
        }
        let mut controls = self.options.sliver;
        controls.maximum_passes = remaining;
        match run_in_state(
            state,
            self.context,
            self.prepared,
            controls,
            self.cancellation,
            QualityConvergenceMode::Coupled,
            rejected,
        ) {
            Ok(repaired) => Ok(RepairOutcome {
                state: repaired.state,
                mutations: repaired.mutations,
                blocked: None,
            }),
            Err(mut failure) => {
                if failure.error.kind == DelaunayVolumeSliverErrorKind::NoAdmissibleOptimization {
                    if let Some(state) = failure.state.take() {
                        // Mutations accepted before the local minimum are part of the
                        // returned state. Any applied change clears the provisional
                        // rejection; an unchanged state carries it.
                        let mutations = std::mem::take(&mut failure.applied);
                        let blocked = mutations.is_empty().then_some(failure.error);
                        return Ok(RepairOutcome {
                            state: *state,
                            mutations,
                            blocked,
                        });
                    }
                }
                Err(self.reject(failure.error, failure.state))
            }
        }
    }
    pub fn reject(
        &self,
        failure: DelaunayVolumeSliverError,
        state: Option<Box<WorkingState<'ctx>>>,
    ) -> DelaunayVolumeMeshError {
        if let Some(state) = state {
            if let Ok(diagnostic) = (*state).into_diagnostic(self.options.sliver, self.cancellation)
            {
                let snapshot = super::super::quality_replay::VolumeQualitySnapshot {
                    topology: &diagnostic.topology,
                    provenance: self.context.provenance,
                    metric: self.context.metric_request,
                    fixed_nodes: self.context.fixed_node_identities,
                    options: self.options,
                };
                if let Err(reason) = self.observer.observe_stalled(snapshot) {
                    tracing::info!(%reason, "stalled quality diagnostic unavailable; original failure preserved");
                }
            }
        }
        sliver_error(failure)
    }
}
