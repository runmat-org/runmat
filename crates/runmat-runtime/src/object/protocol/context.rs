use std::future::Future;

use futures::future::Either;

use crate::context::{ContextFuture, RuntimeContext};

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
/// Select the future directly so an ambient call does not retain another
/// generic async poll frame around the protocol continuation.
pub(super) fn scope<F: Future>(
    runtime: Option<&RuntimeContext>,
    work: F,
) -> Either<ContextFuture<F>, F> {
    match runtime {
        Some(runtime) => Either::Left(runtime.scope(work)),
        None => Either::Right(work),
    }
}
