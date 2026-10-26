use std::future::Future;

use crate::context::RuntimeContext;

// Pass only captures through the caller's frame. Construct the concrete path
// future at its existing exact heap owner, after admission, rather than first
// moving that entire future through each recursive path caller.
#[inline(never)]
pub(super) fn admit_future_with<'a, F, O>(
    runtime: Option<&RuntimeContext>,
    make_work: impl FnOnce() -> F,
) -> crate::execution::RuntimeContinuationFuture<'a, O>
where
    F: Future<Output = Result<O, crate::RuntimeError>> + 'a,
    O: crate::execution::rooted_continuation::RootedCompletion + 'a,
{
    let scoped = crate::context::RuntimeScopeFactory::new(runtime.cloned(), make_work);
    let selected = scoped.runtime().cloned();
    crate::execution::rooted_continuation::admit_rooted_future_with(
        selected,
        "object subscript path",
        crate::builtin::builtin_control_error,
        move |_| scoped.construct(),
    )
}

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Pin work at construction, but snapshot the selected context only on the
/// first poll. An unpolled call must not capture a caller's ambient projections.
pub(super) fn scope<F, O>(
    runtime: Option<&RuntimeContext>,
    work: F,
) -> impl Future<Output = Result<O, crate::RuntimeError>> + use<F, O>
where
    F: Future<Output = Result<O, crate::RuntimeError>>,
{
    crate::context::LazyRuntimeScope::new(runtime.cloned(), work)
}
