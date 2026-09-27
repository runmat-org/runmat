use super::ProviderOperationRecorder;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

thread_local! {
    static ACTIVE: RefCell<Option<ProviderOperationRecorder>> = const { RefCell::new(None) };
}

/// Clone only the current operation's recorder, not any runtime/session handle.
pub fn current_provider_observation() -> Option<ProviderOperationRecorder> {
    ACTIVE.with(|active| active.borrow().clone())
}

struct Restore(Option<ProviderOperationRecorder>);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.0.take());
        });
    }
}

/// Scope one synchronous poll/operation. Never retain activation across await.
/// Explicit None masks an outer operation for deliberately unobserved work.
pub fn with_provider_observation<T>(
    recorder: Option<ProviderOperationRecorder>,
    f: impl FnOnce() -> T,
) -> T {
    let _restore = Restore(ACTIVE.with(|active| active.replace(recorder)));
    f()
}

struct ObservedFuture<F> {
    inner: Option<Pin<Box<F>>>,
    recorder: Option<ProviderOperationRecorder>,
}

impl<F: Future> Future for ObservedFuture<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        with_provider_observation(this.recorder.clone(), || {
            this.inner
                .as_mut()
                .expect("observation future has an inner owner")
                .as_mut()
                .poll(cx)
        })
    }
}

impl<F> Drop for ObservedFuture<F> {
    fn drop(&mut self) {
        let inner = self.inner.take();
        with_provider_observation(self.recorder.clone(), || drop(inner));
    }
}

/// Future-local attribution: nested and interleaved polls restore their caller's
/// scope on Pending, Ready and unwind. This does not own publication/cancellation;
/// retain the unique receipt in the invocation's existing lifetime owner.
pub fn observe_provider_future<F: Future>(
    recorder: Option<ProviderOperationRecorder>,
    future: F,
) -> impl Future<Output = F::Output> {
    ObservedFuture {
        inner: Some(Box::pin(future)),
        recorder,
    }
}
