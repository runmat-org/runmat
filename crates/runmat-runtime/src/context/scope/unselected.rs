//! Explicit original absence, on the same activation stack and future owner.
use super::*;
use std::{marker::PhantomData, rc::Rc};

/// A same-lane proof that absence was selected before any masking/allocation.
/// It is not a RuntimeContext or a managed resource grant.
pub(crate) struct PreparedUnselectedScope {
    _local: PhantomData<Rc<()>>,
}
impl PreparedUnselectedScope {
    pub(crate) fn enter(&self) -> RuntimeContextGuard {
        RuntimeContextGuard::enter_unselected()
    }
    pub(crate) fn prepare(original: Option<&RuntimeContext>) -> Result<Self, RuntimeError> {
        if original.is_some() {
            return Err(crate::build_runtime_error(
                "an explicitly selected runtime cannot be replaced by an unselected scope",
            )
            .with_identifier("RunMat:Runtime:UnselectedScope")
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        super::super::host_allocation::require_unbounded_workspace(
            None,
            "unselected scope",
            "finite context-free poll scopes are unsupported",
        )?;
        Ok(Self {
            _local: PhantomData,
        })
    }
    pub(crate) fn scope<F: Future>(&self, future: F) -> ContextFuture<F> {
        let _guard = RuntimeContextGuard::enter_unselected();
        ContextFuture {
            scope: None,
            future: Some(Box::pin(future)),
        }
    }
}
pub(super) fn poll<F: Future>(
    future: &mut Option<Pin<Box<F>>>,
    cx: &mut Context<'_>,
) -> Poll<F::Output> {
    let _guard = RuntimeContextGuard::enter_unselected();
    let result = future
        .as_mut()
        .expect("unselected future polled after completion")
        .as_mut()
        .poll(cx);
    if result.is_ready() {
        drop(future.take());
    }
    result
}
pub(super) fn drop_future<F>(future: &mut Option<Pin<Box<F>>>) {
    if let Some(future) = future.take() {
        let _guard = RuntimeContextGuard::enter_unselected();
        drop(future);
    }
}
