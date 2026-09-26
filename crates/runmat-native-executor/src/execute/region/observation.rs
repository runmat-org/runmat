use super::super::state::HostState;
use crate::region::{OptimizedRegionPlan, RegionObservation, SiteIdentity};
use crate::NativeExecutorResult;
use runmat_runtime::native::{
    NativeSiteOutcome, NativeSiteOutcomeKind, NativeSitePhase, NativeSiteRequest,
};
use std::collections::BTreeSet;
#[cfg(test)]
mod tests;

pub(crate) struct GenericRegionObservation {
    exit: runmat_types::ProgramPointId,
    remaining: BTreeSet<SiteIdentity>,
    observation: RegionObservation,
}
impl GenericRegionObservation {
    pub(crate) fn new(plan: &OptimizedRegionPlan, observation: RegionObservation) -> Self {
        let mut remaining = plan.skipped_sites.clone();
        remaining.insert(plan.entry);
        Self {
            exit: plan.exit,
            remaining,
            observation,
        }
    }
    fn before(self, site: SiteIdentity, forward_goto: bool, cancelled: bool) -> Option<Self> {
        if cancelled {
            return None;
        }
        if site.point == self.exit {
            if self.remaining.is_empty() {
                self.observation.complete();
            }
            return None;
        }
        if self.remaining.contains(&site) || forward_goto {
            Some(self)
        } else {
            None
        }
    }

    fn after(
        mut self,
        site: SiteIdentity,
        result: &NativeExecutorResult<NativeSiteOutcome>,
    ) -> Option<Self> {
        match result {
            Ok(outcome) if outcome.kind == NativeSiteOutcomeKind::CONTINUE => {
                self.remaining.remove(&site);
                Some(self)
            }
            Ok(outcome)
                if site.phase == runmat_native_codegen::NativeSitePhase::Terminator
                    && outcome.kind == NativeSiteOutcomeKind::EDGE
                    && outcome.edge == 0 =>
            {
                Some(self)
            }
            _ => None,
        }
    }
}
fn identity(request: NativeSiteRequest) -> SiteIdentity {
    SiteIdentity {
        point: runmat_types::ProgramPointId {
            function: runmat_types::ProgramFunctionId(request.function),
            block: request.block,
            position: request.position,
        },
        phase: match request.phase {
            NativeSitePhase::RVALUE => runmat_native_codegen::NativeSitePhase::Rvalue,
            NativeSitePhase::STATEMENT => runmat_native_codegen::NativeSitePhase::Statement,
            NativeSitePhase::TERMINATOR_RVALUE => {
                runmat_native_codegen::NativeSitePhase::TerminatorRvalue
            }
            _ => runmat_native_codegen::NativeSitePhase::Terminator,
        },
        ordinal: request.ordinal,
    }
}
pub(in crate::execute) fn before_site(state: &mut HostState, request: NativeSiteRequest) {
    let Some(pending) = state.region_observation.take() else {
        return;
    };
    let forward_goto = request.phase == NativeSitePhase::TERMINATOR
        && request.function == state.function.id.0
        && state
            .function
            .blocks
            .iter()
            .find(|block| block.id.0 == request.block)
            .is_some_and(|block| match &block.terminator.kind {
                runmat_native_codegen::NativeTerminatorKind::Goto { edge } => pending
                    .remaining
                    .iter()
                    .any(|site| site.point.block == edge.target.0),
                _ => false,
            });
    state.region_observation = pending.before(
        identity(request),
        forward_goto,
        state
            .runtime
            .cancellation()
            .load(std::sync::atomic::Ordering::Relaxed),
    );
}
pub(in crate::execute) fn after_site(
    state: &mut HostState,
    request: NativeSiteRequest,
    result: &NativeExecutorResult<NativeSiteOutcome>,
) {
    if let Some(pending) = state.region_observation.take() {
        state.region_observation = pending.after(identity(request), result);
    }
}
