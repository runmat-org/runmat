//! Future-local opt-in scope for one explicitly prepared resource transcript.
//! Late resource retirement does not consult this ambient slot: issued tickets
//! retain the canonical receipt identity themselves.
use super::ProviderResourceScope;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

thread_local! {
    static ACTIVE: RefCell<Option<ProviderResourceScope>> = const { RefCell::new(None) };
}

pub fn current_provider_resource_scope() -> Option<ProviderResourceScope> {
    ACTIVE.with(|active| active.borrow().clone())
}

struct Restore(Option<ProviderResourceScope>);
impl Drop for Restore {
    fn drop(&mut self) {
        ACTIVE.with(|active| {
            active.replace(self.0.take());
        });
    }
}

fn with_scope<T>(scope: Option<ProviderResourceScope>, f: impl FnOnce() -> T) -> T {
    let _restore = Restore(ACTIVE.with(|active| active.replace(scope)));
    f()
}

struct ScopedFuture<F> {
    inner: Option<Pin<Box<F>>>,
    scope: Option<ProviderResourceScope>,
}

impl<F: Future> Future for ScopedFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        with_scope(this.scope.clone(), || {
            this.inner
                .as_mut()
                .expect("resource scope future has inner owner")
                .as_mut()
                .poll(cx)
        })
    }
}

impl<F> Drop for ScopedFuture<F> {
    fn drop(&mut self) {
        let inner = self.inner.take();
        with_scope(self.scope.clone(), || drop(inner));
    }
}

pub fn observe_provider_resource_future<F: Future>(
    scope: Option<ProviderResourceScope>,
    future: F,
) -> impl Future<Output = F::Output> {
    ScopedFuture {
        inner: Some(Box::pin(future)),
        scope,
    }
}

#[cfg(test)]
#[path = "resource_activation/tests.rs"]
mod tests;
