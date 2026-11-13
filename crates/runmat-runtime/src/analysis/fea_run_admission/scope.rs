use super::lifecycle::RunState;
use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

thread_local! {
    static CURRENT: RefCell<Option<Arc<RunState>>> = const { RefCell::new(None) };
}

pub(super) fn current() -> Option<Arc<RunState>> {
    CURRENT.with(|v| v.borrow().clone())
}

pub(super) fn run<T>(state: Arc<RunState>, operation: impl FnOnce() -> T) -> T {
    let _usage = state.accounting.enter();
    struct Restore(Option<Arc<RunState>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|v| *v.borrow_mut() = self.0.take());
        }
    }
    let previous = CURRENT.with(|v| v.replace(Some(state)));
    let _restore = Restore(previous);
    let _interrupt =
        current().map(|state| crate::interrupt::replace_interrupt(Some(state.token())));
    operation()
}

/// Installs a run only while its future is being polled. Unrelated tasks on the
/// same executor thread cannot inherit authorization while this future waits.
pub struct ScopedFeaRunFuture<F> {
    pub(super) state: Arc<RunState>,
    pub(super) future: Pin<Box<F>>,
}

impl<F: Future> Future for ScopedFeaRunFuture<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _ = this.state.check();
        run(Arc::clone(&this.state), || {
            this.future.as_mut().poll(context)
        })
    }
}

/// Lease-owned async boundary. A lost hold wakes and refuses this run even
/// when the host future has not produced another value or runtime checkpoint.
pub struct CheckedFeaRunFuture<F> {
    pub(super) state: Arc<RunState>,
    pub(super) future: Pin<Box<F>>,
    pub(super) cancelled: runmat_execution::cancellation::Cancelled,
}
impl<F: Future> Future for CheckedFeaRunFuture<F> {
    type Output = Result<F::Output, super::FeaRunExecutionError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Err(denied) = this.state.check() {
            return Poll::Ready(Err(super::FeaRunExecutionError::Denied(denied)));
        }
        if Pin::new(&mut this.cancelled).poll(context).is_ready() {
            return Poll::Ready(Err(match this.state.check() {
                Err(denied) => super::FeaRunExecutionError::Denied(denied),
                Ok(()) => super::FeaRunExecutionError::Cancelled,
            }));
        }
        let result = run(Arc::clone(&this.state), || {
            this.future.as_mut().poll(context)
        });
        if let Err(denied) = this.state.check() {
            return Poll::Ready(Err(super::FeaRunExecutionError::Denied(denied)));
        }
        if this.state.token().is_cancelled() {
            return Poll::Ready(Err(super::FeaRunExecutionError::Cancelled));
        }
        result.map(Ok)
    }
}
