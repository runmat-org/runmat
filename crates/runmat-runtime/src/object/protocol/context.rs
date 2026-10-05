use std::future::Future;

use crate::context::RuntimeContext;

/// Keep session-owned metadata and callbacks on the selected context for
/// every poll, including the first admission and the future's destructor.
pub(super) async fn scope<F: Future>(runtime: Option<&RuntimeContext>, work: F) -> F::Output {
    match runtime {
        Some(runtime) => runtime.scope(work).await,
        None => work.await,
    }
}
