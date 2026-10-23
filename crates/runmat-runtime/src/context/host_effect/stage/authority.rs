//! Narrow runtime authority retained by one staged host effect.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use runmat_execution::cancellation::CancellationToken;
use runmat_execution::ProgramRevision;

use crate::context::{
    RuntimeContext, RuntimeContextState, RuntimeHostEffectResourceService, RuntimeParallelService,
    RuntimePlacementService,
};
use crate::execution::RuntimeExecutionServices;

/// Strongly retain only the session and service owners whose identity is part
/// of staged-effect admission. Poll-local source, call, output, console, and
/// debug projections remain owned by the actual continuation context.
pub(in crate::context) struct HostEffectStageAuthority {
    state: Rc<RuntimeContextState>,
    live_cancellation: Rc<RefCell<CancellationToken>>,
    execution: Rc<dyn RuntimeExecutionServices>,
    program_revision: Option<Arc<ProgramRevision>>,
    placement: Option<Rc<dyn RuntimePlacementService>>,
    host_effect_resources: Option<Rc<dyn RuntimeHostEffectResourceService>>,
    parallel: Option<Rc<dyn RuntimeParallelService>>,
}

impl HostEffectStageAuthority {
    pub(in crate::context) fn new(
        state: Rc<RuntimeContextState>,
        live_cancellation: Rc<RefCell<CancellationToken>>,
        execution: Rc<dyn RuntimeExecutionServices>,
        program_revision: Option<Arc<ProgramRevision>>,
        placement: Option<Rc<dyn RuntimePlacementService>>,
        host_effect_resources: Option<Rc<dyn RuntimeHostEffectResourceService>>,
        parallel: Option<Rc<dyn RuntimeParallelService>>,
    ) -> Self {
        Self {
            state,
            live_cancellation,
            execution,
            program_revision,
            placement,
            host_effect_resources,
            parallel,
        }
    }

    pub(super) fn original_cancellation_is_live(&self, original: &CancellationToken) -> bool {
        original.same_generation(&self.live_cancellation.borrow())
    }

    pub(super) fn parallel(&self) -> Option<&Rc<dyn RuntimeParallelService>> {
        self.parallel.as_ref()
    }

    pub(super) fn matches_runtime(
        &self,
        runtime: &RuntimeContext,
        original_cancellation: &CancellationToken,
    ) -> bool {
        self.state.local_identity == runtime.local_identity()
            && same(self.placement.as_ref(), runtime.service_ports().placement())
            && same(
                self.host_effect_resources.as_ref(),
                runtime.service_ports().host_effect_resources(),
            )
            && same(self.parallel.as_ref(), runtime.service_ports().parallel())
            && Rc::ptr_eq(&self.execution, runtime.execution())
            && self.program_revision.as_deref() == runtime.program_revision()
            && original_cancellation.same_generation(&runtime.cancellation())
    }
}

fn same<T: ?Sized>(left: Option<&Rc<T>>, right: Option<&Rc<T>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => Rc::ptr_eq(left, right),
        (None, None) => true,
        _ => false,
    }
}
