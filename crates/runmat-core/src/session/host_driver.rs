//! One native Tokio reactor owned by the application host, shared by HTTP and
//! process services even when Core is polled by a non-Tokio executor.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(super) fn host_driver() -> Result<&'static tokio::runtime::Runtime, HostDriverInitError> {
    use std::sync::OnceLock;

    static DRIVER: OnceLock<Result<tokio::runtime::Runtime, HostDriverInitError>> = OnceLock::new();
    DRIVER
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .map_err(|error| HostDriverInitError(std::sync::Arc::new(error)))
        })
        .as_ref()
        .map_err(Clone::clone)
}

#[derive(Debug, Clone)]
pub(super) struct HostDriverInitError(pub(super) std::sync::Arc<std::io::Error>);

impl std::fmt::Display for HostDriverInitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.0.as_ref(), formatter)
    }
}

impl std::error::Error for HostDriverInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

/// Poll and drop one reactor-dependent future under the installed host driver.
/// No task is detached: cancelling the caller drops the same owned future.
pub(super) struct EnteredHostFuture<F> {
    handle: tokio::runtime::Handle,
    inner: Option<Pin<Box<F>>>,
}

impl<F: Future> EnteredHostFuture<F> {
    pub(super) fn new(handle: tokio::runtime::Handle, future: F) -> Self {
        Self {
            handle,
            inner: Some(Box::pin(future)),
        }
    }
}

impl<F: Future> Future for EnteredHostFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _entered = this.handle.enter();
        this.inner
            .as_mut()
            .expect("host future retained")
            .as_mut()
            .poll(context)
    }
}

impl<F> Drop for EnteredHostFuture<F> {
    fn drop(&mut self) {
        let _entered = self.handle.enter();
        drop(self.inner.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    struct PendingDrop(Rc<Cell<bool>>);

    impl Future for PendingDrop {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            assert!(tokio::runtime::Handle::try_current().is_ok());
            Poll::Pending
        }
    }

    impl Drop for PendingDrop {
        fn drop(&mut self) {
            self.0.set(tokio::runtime::Handle::try_current().is_ok());
        }
    }

    #[test]
    fn ordinary_executor_enters_host_driver_for_poll_and_drop() {
        assert!(tokio::runtime::Handle::try_current().is_err());
        let entered = Rc::new(Cell::new(false));
        let driver = host_driver().expect("native driver");
        let mut future = Box::pin(EnteredHostFuture::new(
            driver.handle().clone(),
            PendingDrop(entered.clone()),
        ));
        let waker = futures::task::noop_waker();
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        drop(future);
        assert!(entered.get());
        assert!(tokio::runtime::Handle::try_current().is_err());
    }
}
