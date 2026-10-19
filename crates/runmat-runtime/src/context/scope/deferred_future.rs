//! Queue-owned construction trampoline for an admitted poll-scoped future.

use super::{ContextFuture, PreparedRuntimeFutureScope};
use crate::RuntimeError;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Owns an admitted factory until the process driver has unwound the caller's
/// poll stack. Construction and polling still happen on the same OS thread;
/// the first poll only materializes `F`, then wakes a distinct poll for `F`.
pub struct DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
    state: DeferredState<F, M>,
    _output: PhantomData<fn() -> T>,
}

enum DeferredState<F, M> {
    Prepared {
        scope: PreparedRuntimeFutureScope<F>,
        make_future: M,
    },
    Running(ContextFuture<F>),
    Finished,
}

impl<F, M, T> DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
    pub(super) fn new(prepared: PreparedRuntimeFutureScope<F>, make_future: M) -> Self {
        Self {
            state: DeferredState::Prepared {
                scope: prepared,
                make_future,
            },
            _output: PhantomData,
        }
    }
}

impl<F, M, T> Future for DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
    type Output = Result<T, RuntimeError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        if let DeferredState::Running(running) = &mut this.state {
            return Pin::new(running).poll(context);
        }

        let (scope, make_future) = match std::mem::replace(&mut this.state, DeferredState::Finished)
        {
            DeferredState::Prepared { scope, make_future } => (scope, make_future),
            DeferredState::Running(_) => unreachable!("running state returned above"),
            DeferredState::Finished => {
                panic!("deferred runtime future polled after completion")
            }
        };
        match scope.try_scope_with(make_future) {
            Ok(running) => {
                this.state = DeferredState::Running(running);
                // Do not stack the child's first poll on its construction
                // frame. The driver may service this wake immediately, but
                // only after this call has returned and unwound.
                context.waker().wake_by_ref();
                Poll::Pending
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl<F, M, T> Drop for DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
    fn drop(&mut self) {
        if let DeferredState::Prepared { scope, make_future } =
            std::mem::replace(&mut self.state, DeferredState::Finished)
        {
            scope.drop_unconstructed(make_future);
        }
    }
}

// Neither the factory nor the prepared scope is address-sensitive, and the
// constructed ContextFuture pins F in its own Box.
impl<F, M, T> Unpin for DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
}
