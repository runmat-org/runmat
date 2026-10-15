use std::future::Future;

use crate::context::RuntimeContext;

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Pin work at construction, but snapshot the selected context only on the
/// first poll. An unpolled call must not capture a caller's ambient projections.
pub(super) fn scope<'runtime, F: Future>(
    runtime: Option<&'runtime RuntimeContext>,
    work: F,
) -> impl Future<Output = F::Output> + use<'runtime, F> {
    let work = Box::pin(work);
    async move {
        match runtime {
            Some(runtime) => runtime.scope_pinned(work).await,
            None => work.await,
        }
    }
}
