use std::future::Future;
use std::pin::Pin;

use futures::future::Either;

use crate::context::{ContextFuture, RuntimeContext};

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Pin the selected work once so neither path retains its potentially large
/// continuation inline in an enclosing object or builtin future.
pub(super) fn scope<F: Future>(
    runtime: Option<&RuntimeContext>,
    work: F,
) -> Either<ContextFuture<F>, Pin<Box<F>>> {
    let work = Box::pin(work);
    match runtime {
        Some(runtime) => Either::Left(runtime.scope_pinned(work)),
        None => Either::Right(work),
    }
}
