//! Explicit per-invocation host selection. Installing this context does not
//! acquire a lease: ordinary scripts remain unmetered until an FEA root starts.
use super::FeaRunAdmission;
use std::{
    cell::RefCell,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

thread_local! {
    static HOST: RefCell<Option<Arc<dyn FeaRunAdmission>>> = const { RefCell::new(None) };
}

pub(super) fn current() -> Option<Arc<dyn FeaRunAdmission>> {
    HOST.with(|value| value.borrow().clone())
}

#[derive(Clone)]
pub struct FeaRunAdmissionContext {
    admission: Arc<dyn FeaRunAdmission>,
}

impl FeaRunAdmissionContext {
    pub fn new(admission: Arc<dyn FeaRunAdmission>) -> Self {
        Self { admission }
    }

    pub fn run_sync<T>(&self, operation: impl FnOnce() -> T) -> T {
        struct Restore(Option<Arc<dyn FeaRunAdmission>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                HOST.with(|value| *value.borrow_mut() = self.0.take());
            }
        }
        let previous = HOST.with(|value| value.replace(Some(Arc::clone(&self.admission))));
        let _restore = Restore(previous);
        operation()
    }

    /// Host identity is present only during this invocation's polls, never
    /// while an unrelated window/task runs on the same executor thread.
    pub fn run_future<F: Future>(&self, future: F) -> FeaRunAdmissionFuture<F> {
        FeaRunAdmissionFuture {
            context: self.clone(),
            future: Box::pin(future),
        }
    }
}

pub struct FeaRunAdmissionFuture<F> {
    context: FeaRunAdmissionContext,
    future: Pin<Box<F>>,
}

impl<F: Future> Future for FeaRunAdmissionFuture<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.context.run_sync(|| this.future.as_mut().poll(context))
    }
}
