//! Exact ownership of a selected instruction's root-free control completion.
use super::{DispatchDecision, DispatchHandled, DispatchResult};
use futures::FutureExt;
use runmat_gc::{Trace, Tracer};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::{RootedResult, RuntimeContinuationFuture};
use runmat_runtime::RuntimeError;
use std::future::Future;

pub(in crate::interpreter::dispatch) type ControlDispatchFuture<'a> =
    Result<RuntimeContinuationFuture<'a, Option<RootedResult<DispatchHandled>>>, RuntimeError>;

impl Trace for DispatchDecision {
    fn trace(&self, _tracer: &mut dyn Tracer) {
        // Explicit unit patterns make a new or payload-bearing variant choose
        // its actual tracing policy rather than silently acquiring no roots.
        match self {
            Self::ContinueLoop | Self::FallThrough | Self::Return => {}
        }
    }
}

impl Trace for DispatchHandled {
    fn trace(&self, tracer: &mut dyn Tracer) {
        match self {
            Self::Generic(decision) | Self::ReturnValue(decision) | Self::Return(decision) => {
                decision.trace(tracer);
            }
        }
    }
}

pub(super) fn root_control(
    result: DispatchResult,
) -> Result<Option<RootedResult<DispatchHandled>>, RuntimeError> {
    result?
        .map(RootedResult::new)
        .transpose()
        .map_err(|error| error.into_runtime_error())
}

#[inline(never)]
pub(in crate::interpreter::dispatch) fn admit<'a, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> ControlDispatchFuture<'a>
where
    F: Future<Output = DispatchResult> + 'a,
{
    // The mapped concrete future is inside the original single allocation.
    // No root registry/lease backing is allocated for these exhaustive control
    // enums: the canonical root planner's zero-handle path is an empty lease.
    runmat_runtime::execution::admit_optional_rooted_result_with(runtime, move || {
        make_future().map(root_control)
    })
}
