//! Direct proposals are prepared once. A coupled pass consumes their deferred
//! branches only when the checked local families have no improving step.
use super::*;
pub(super) struct DeferredSearch {
    pub(super) seeds: Vec<WorkingSeed>,
    pub(super) baseline: SpectrumDelta,
    pub(super) policy: SearchPolicy,
}
impl WorkingReconnectionSearch {
    pub(in crate::cdt::volume_sliver) fn complete<'ctx>(
        &mut self,
        state: &mut impl SearchState<'ctx>,
        cancellation: &dyn MeshingCancellationSignal,
        work: &mut OptimizationWork<'_>,
    ) -> Result<Option<WorkingSeed>, DelaunayVolumeSliverError> {
        let Some(pending) = self.pending.take() else {
            return Ok(None);
        };
        lookahead::search(
            state,
            pending.seeds,
            &pending.baseline,
            cancellation,
            work,
            pending.policy,
            &mut self.rejection_counts,
        )
    }
}
