use super::super::state::HostState;
#[cfg(test)]
use crate::region::OptimizedRegionPlan;
use crate::region::{OptimizedRegionPlanLease, RegionObservation, SiteIdentity};
use crate::NativeExecutorResult;
use runmat_runtime::native::{
    NativeSiteOutcome, NativeSiteOutcomeKind, NativeSitePhase, NativeSiteRequest,
};
#[cfg(test)]
mod native_original_tests;
#[cfg(test)]
pub(in crate::execute) mod test_support;
#[cfg(test)]
mod tests;

pub(crate) struct GenericRegionObservation {
    exit: SiteIdentity,
    plan: OptimizedRegionPlanLease,
    cursor: usize,
    observation: RegionObservation,
    host: Option<runmat_runtime::numeric_region::service::NativeNumericRegionHostExecution>,
}
impl GenericRegionObservation {
    pub(crate) fn new(plan: OptimizedRegionPlanLease, observation: RegionObservation) -> Self {
        Self {
            exit: plan.exit_site,
            plan,
            cursor: 0,
            observation,
            host: None,
        }
    }
    pub(crate) fn with_host(
        mut self,
        host: runmat_runtime::numeric_region::service::NativeNumericRegionHostExecution,
    ) -> Self {
        self.host = Some(host);
        self
    }
    fn before(self, site: SiteIdentity, cancelled: bool) -> Option<Self> {
        if cancelled {
            return None;
        }
        if site == self.exit {
            if self.cursor == self.plan.traversal_sites.len()
                && self
                    .host
                    .as_ref()
                    .is_none_or(|host| host.route().complete(self.observation.runtime()))
            {
                self.observation.complete();
            }
            return None;
        }
        if self.plan.traversal_sites.get(self.cursor) == Some(&site) {
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
                self.cursor += 1;
                Some(self)
            }
            Ok(outcome)
                if site.phase == runmat_native_codegen::NativeSitePhase::Terminator
                    && outcome.kind == NativeSiteOutcomeKind::EDGE
                    && outcome.edge == 0 =>
            {
                self.cursor += 1;
                Some(self)
            }
            _ => None,
        }
    }
}

pub(in crate::execute) fn host_operand(
    state: &HostState,
    site: NativeSiteRequest,
    operator: runmat_types::OperatorKind,
    operand: usize,
) -> NativeExecutorResult<Option<&runmat_value::Value>> {
    let Some(host) = state
        .region_observation
        .as_ref()
        .and_then(|pending| pending.host.as_ref())
    else {
        return Ok(None);
    };
    host.route().begin(&state.runtime, site, operator)?;
    let Some(index) = host.route().input_substitution(site, operand) else {
        return Ok(None);
    };
    host.input(index).filter(|value| matches!(value, runmat_value::Value::Num(_))
        || matches!(value, runmat_value::Value::Tensor(tensor) if matches!(tensor.numeric_dtype(), runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64)))
        .map(Some).ok_or_else(|| {
        crate::NativeExecutorError::Host(
            "native host input substitution exceeds captured inputs".into(),
        )
    })
}

pub(in crate::execute) fn host_route(
    state: &HostState,
) -> Option<runmat_runtime::context::NativeRegionHostRoute> {
    state
        .region_observation
        .as_ref()
        .and_then(|pending| pending.host.as_ref())
        .map(|host| host.route().clone())
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
pub(in crate::execute) fn before_site(
    state: &mut HostState,
    request: NativeSiteRequest,
) -> NativeExecutorResult<()> {
    super::super::call_suspension::numeric_region::take_original_completion(state, request)?;
    let Some(mut pending) = state.region_observation.take() else {
        return Ok(());
    };
    if !state.runtime.is_cancelled()
        && identity(request) == pending.exit
        && pending.cursor == pending.plan.traversal_sites.len()
    {
        if let Some(future) = pending.observation.finish_original() {
            return super::super::call_suspension::numeric_region::begin_original_completion(
                state, request, future,
            );
        }
    }
    state.region_observation = pending.before(identity(request), state.runtime.is_cancelled());
    Ok(())
}
pub(in crate::execute) fn after_site(
    state: &mut HostState,
    request: NativeSiteRequest,
    result: &NativeExecutorResult<NativeSiteOutcome>,
) -> NativeExecutorResult<()> {
    if let Some(pending) = state.region_observation.take() {
        // The exact hostcall suspends before this source site has completed.
        // Retain its cursor and audit, advancing only after the resumed site
        // publishes its ordinary result.
        let matching_suspend = state
            .pending_call
            .as_ref()
            .is_some_and(|call| call.request == request)
            && matches!(result, Ok(outcome) if outcome.kind == NativeSiteOutcomeKind::EXIT);
        state.region_observation = if matching_suspend {
            Some(pending)
        } else {
            if matches!(result, Ok(outcome) if outcome.kind == NativeSiteOutcomeKind::CONTINUE) {
                if let Some(host) = &pending.host {
                    host.route().finish_if_pending(request)?;
                }
            }
            pending.after(identity(request), result)
        };
    }
    Ok(())
}

pub(in crate::execute) fn active_audit(
    state: &HostState,
) -> Option<runmat_runtime::numeric_region::service::SourceRegionRouteAudit> {
    state
        .region_observation
        .as_ref()
        .and_then(|pending| pending.observation.original_audit())
}
