//! Large controller phases use the existing exact-layout continuation owner.
//! The factory captures only original arguments/borrows, not a constructed
//! phase future. Domain inputs and results retain their original root owners.
use super::super::ScalarCooperation;
use crate::{context::RuntimeContext, BuiltinResult};
use std::future::Future;

pub(in crate::builtins::math::optim::ga) fn original(
    cooperation: &ScalarCooperation,
) -> BuiltinResult<RuntimeContext> {
    cooperation.runtime().cloned().ok_or_else(|| {
        super::super::callbacks::unavailable(
            "genetic controller phases require their selected execution context",
        )
    })
}

pub(in crate::builtins::math::optim::ga) fn run<'a, T, F>(
    runtime: &RuntimeContext,
    make_phase: impl FnOnce() -> F,
) -> BuiltinResult<crate::execution::RuntimeContinuationFuture<'a, T>>
where
    T: 'a,
    F: Future<Output = BuiltinResult<T>> + 'a,
{
    crate::execution::admit_native_hostcall_with_contextual_error(runtime, make_phase)
}
