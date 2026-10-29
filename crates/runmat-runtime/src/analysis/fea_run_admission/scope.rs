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
    struct Restore(Option<Arc<RunState>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|v| *v.borrow_mut() = self.0.take());
        }
    }
    let previous = CURRENT.with(|v| v.replace(Some(state)));
    let _restore = Restore(previous);
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
        run(Arc::clone(&this.state), || {
            this.future.as_mut().poll(context)
        })
    }
}
