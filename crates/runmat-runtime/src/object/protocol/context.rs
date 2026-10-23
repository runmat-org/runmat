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
    crate::execution::rooted_continuation::admit_rooted_future_with(
        runtime.cloned(),
        "object subscript path",
        crate::builtin::builtin_control_error,
        move |_| make_work(),
    )
}

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Pin work at construction, but snapshot the selected context only on the
/// first poll. An unpolled call must not capture a caller's ambient projections.
pub(super) async fn scope<F, O>(
    runtime: Option<&RuntimeContext>,
    work: F,
) -> Result<O, crate::RuntimeError>
where
    F: Future<Output = Result<O, crate::RuntimeError>>,
{
    match runtime {
        Some(runtime) => {
            runtime
                .prepare_poll_scope()?
                .prepare_future::<F>()?
                .try_scope_with(|| work)?
                .await
        }
        None => Box::pin(work).await,
    }
}
