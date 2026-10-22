use std::future::Future;

use crate::context::RuntimeContext;

pub(super) fn admit_future<'a, F, O>(
    runtime: Option<&RuntimeContext>,
    work: F,
) -> crate::execution::RuntimeContinuationFuture<'a, O>
where
    F: Future<Output = Result<O, crate::RuntimeError>> + 'a,
    O: crate::execution::rooted_continuation::RootedCompletion + 'a,
{
    crate::execution::rooted_continuation::admit_rooted_future(
        runtime,
        "object subscript path",
        crate::builtin::builtin_control_error,
        work,
    )
}

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Pin work at construction, but snapshot the selected context only on the
/// first poll. An unpolled call must not capture a caller's ambient projections.
pub(super) fn scope<'runtime, F, O>(
    runtime: Option<&'runtime RuntimeContext>,
    work: F,
) -> impl Future<Output = Result<O, crate::RuntimeError>> + use<'runtime, F, O>
where
    F: Future<Output = Result<O, crate::RuntimeError>>,
{
    async move {
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
}
