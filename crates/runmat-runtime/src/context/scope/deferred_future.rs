//! Queue-owned construction trampoline for an admitted poll-scoped future.

use super::{FuturePreparation, PollFutureScope, PreparedRuntimeFutureScope};
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
    scope: PollFutureScope,
    preparation: Option<FuturePreparation>,
    state: DeferredState<F, M>,
    _output: PhantomData<fn() -> T>,
}

enum DeferredState<F, M> {
    Prepared(M),
    Running(Option<Pin<Box<F>>>),
    Finished,
}

impl<F, M, T> DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
    pub(super) fn new(prepared: PreparedRuntimeFutureScope<F>, make_future: M) -> Self {
        let PreparedRuntimeFutureScope {
            scope,
            preparation,
            _future: _,
        } = prepared;
        Self {
            scope,
            preparation: Some(preparation),
            state: DeferredState::Prepared(make_future),
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
            return this.scope.poll(running, context);
        }

        let make_future = match std::mem::replace(&mut this.state, DeferredState::Finished) {
            DeferredState::Prepared(make_future) => make_future,
            DeferredState::Running(_) => unreachable!("running state returned above"),
            DeferredState::Finished => {
                panic!("deferred runtime future polled after completion")
            }
        };
        let preparation = this.preparation.take().expect("prepared future admission");
        match preparation.try_construct(&this.scope, make_future) {
            Ok(running) => {
                this.state = DeferredState::Running(Some(running));
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
        match std::mem::replace(&mut self.state, DeferredState::Finished) {
            DeferredState::Prepared(make_future) => {
                self.preparation
                    .take()
                    .expect("prepared future admission")
                    .drop_unconstructed(&mut self.scope, make_future);
            }
            DeferredState::Running(mut running) => self.scope.drop_future(&mut running),
            DeferredState::Finished => {}
        }
    }
}

// Neither the factory nor the prepared scope is address-sensitive, and the
// constructed future pins F in its original admitted Box.
impl<F, M, T> Unpin for DeferredRuntimeFuture<F, M, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
    M: FnOnce() -> F,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use std::{cell::Cell, rc::Rc};

    #[test]
    fn lifecycle_state_does_not_repeat_the_poll_authority() {
        type Ready = std::future::Ready<Result<(), RuntimeError>>;
        assert!(
            std::mem::size_of::<DeferredState<Ready, fn() -> Ready>>()
                <= 2 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn factory_and_constructed_future_retire_under_the_selected_context() {
        struct Probe {
            context: RuntimeContext,
            dropped: Rc<Cell<bool>>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                let active = crate::context::legacy::active().expect("selected drop scope");
                assert_eq!(active.state_identity(), self.context.state_identity());
                self.dropped.set(true);
            }
        }

        for construct in [false, true] {
            let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let dropped = Rc::new(Cell::new(false));
            let constructed = Rc::new(Cell::new(false));
            let probe = Probe {
                context: runtime.clone(),
                dropped: dropped.clone(),
            };
            let constructed_in_factory = constructed.clone();
            let make_future = move || {
                let active = crate::context::legacy::active().expect("selected factory scope");
                assert_eq!(active.state_identity(), probe.context.state_identity());
                constructed_in_factory.set(true);
                async move {
                    let _probe = probe;
                    std::future::pending::<Result<(), RuntimeError>>().await
                }
            };
            let mut deferred = runtime
                .prepare_poll_scope()
                .unwrap()
                .prepare_future::<_>()
                .unwrap()
                .defer_result(make_future);
            if construct {
                let waker = futures::task::noop_waker();
                let mut context = Context::from_waker(&waker);
                assert!(Pin::new(&mut deferred).poll(&mut context).is_pending());
                assert!(constructed.get());
                assert!(!dropped.get());
            }
            drop(deferred);
            assert_eq!(constructed.get(), construct);
            assert!(dropped.get());
        }
    }
}
