use super::super::state::HostState;
use crate::region::{OptimizedRegionPlan, RegionObservation, SiteIdentity};
use crate::NativeExecutorResult;
use runmat_runtime::native::{
    NativeSiteOutcome, NativeSiteOutcomeKind, NativeSitePhase, NativeSiteRequest,
};
use std::collections::VecDeque;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(in crate::execute) mod test_support;

pub(crate) struct GenericRegionObservation {
    exit: runmat_types::ProgramPointId,
    remaining: VecDeque<SiteIdentity>,
    observation: RegionObservation,
}
impl GenericRegionObservation {
    pub(crate) fn new(plan: &OptimizedRegionPlan, observation: RegionObservation) -> Self {
        Self {
            exit: plan.exit,
            remaining: plan.traversal_sites.iter().copied().collect(),
            observation,
        }
    }
    fn before(self, site: SiteIdentity, cancelled: bool) -> Option<Self> {
        if cancelled {
            return None;
        }
        if site.point == self.exit {
            if self.remaining.is_empty() {
                self.observation.complete();
            }
            return None;
        }
        if self.remaining.front() == Some(&site) {
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
                self.remaining.pop_front();
                Some(self)
            }
            Ok(outcome)
                if site.phase == runmat_native_codegen::NativeSitePhase::Terminator
                    && outcome.kind == NativeSiteOutcomeKind::EDGE
                    && outcome.edge == 0 =>
            {
                self.remaining.pop_front();
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
    state.region_observation = pending.before(
        identity(request),
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
