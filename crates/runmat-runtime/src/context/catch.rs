//! Poll-scoped active catch authority for dynamic calls and rethrow.

use super::{RuntimeContext, RuntimeContextLocalId};
use runmat_thread_local::runmat_thread_local;
use runmat_value::MException;
use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

runmat_thread_local! {
    static ACTIVE_CATCHES: RefCell<Vec<ActiveCatchFrame>> = const { RefCell::new(Vec::new()) };
}

struct ActiveCatchFrame {
    token: u64,
    owner: RuntimeContextLocalId,
    exception: Option<Rc<MException>>,
}

#[must_use]
/// A thread-local scope that cannot cross threads.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<runmat_runtime::context::ActiveCatchGuard>();
/// ```
pub struct ActiveCatchGuard {
    token: u64,
    _local_affinity: PhantomData<Rc<()>>,
}

impl ActiveCatchGuard {
    pub(super) fn enter(owner: RuntimeContextLocalId, exception: Option<Rc<MException>>) -> Self {
        let token = ACTIVE_CATCHES.with(|stack| {
            let mut stack = stack.borrow_mut();
            let mut token = stack.last().map_or(1, |frame| frame.token.wrapping_add(1));
            while stack.iter().any(|frame| frame.token == token) {
                token = token.wrapping_add(1);
            }
            stack.push(ActiveCatchFrame {
                token,
                owner,
                exception,
            });
            token
        });
        Self {
            token,
            _local_affinity: PhantomData,
        }
    }
}

impl Drop for ActiveCatchGuard {
    fn drop(&mut self) {
        ACTIVE_CATCHES.with(|stack| {
            let mut stack = stack.borrow_mut();
            let position = stack
                .iter()
                .rposition(|frame| frame.token == self.token)
                .expect("active catch guard must own its stack frame");
            stack.remove(position);
        });
    }
}

/// Poll/drop scope tied to its originating thread-local runtime.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<runmat_runtime::context::ActiveCatchFuture<std::future::Ready<()>>>();
/// ```
pub struct ActiveCatchFuture<F> {
    owner: RuntimeContextLocalId,
    exception: Option<Rc<MException>>,
    // An active catch scope is bound to this thread-local runtime owner.
    _local_affinity: PhantomData<Rc<()>>,
    future: Option<Pin<Box<F>>>,
}

impl<F> ActiveCatchFuture<F> {
    pub(super) fn new(runtime: &RuntimeContext, exception: Option<MException>, future: F) -> Self {
        Self::new_shared(runtime, exception.map(Rc::new), future)
    }

    pub(super) fn new_shared(
        runtime: &RuntimeContext,
        exception: Option<Rc<MException>>,
        future: F,
    ) -> Self {
        Self {
            owner: runtime.local_identity(),
            exception,
            _local_affinity: PhantomData,
            future: Some(Box::pin(future)),
        }
    }
}

impl<F: Future> Future for ActiveCatchFuture<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let _guard = ActiveCatchGuard::enter(this.owner, this.exception.clone());
        let result = this
            .future
            .as_mut()
            .expect("catch future polled after completion")
            .as_mut()
            .poll(cx);
        if result.is_ready() {
            drop(this.future.take());
        }
        result
    }
}

impl<F> Drop for ActiveCatchFuture<F> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let _guard = ActiveCatchGuard::enter(self.owner, self.exception.clone());
            drop(future);
        }
    }
}

impl<F> Unpin for ActiveCatchFuture<F> {}

pub(super) fn active_for(runtime: &RuntimeContext) -> Option<MException> {
    active_shared_for(runtime).as_deref().cloned()
}

pub(super) fn active_shared_for(runtime: &RuntimeContext) -> Option<Rc<MException>> {
    ACTIVE_CATCHES.with(|stack| {
        stack
            .borrow()
            .iter()
            .rev()
            .find(|frame| frame.owner == runtime.local_identity())
            .and_then(|frame| frame.exception.clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CheckPending {
        runtime: RuntimeContext,
        polls: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
    }

    impl Future for CheckPending {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            assert_eq!(
                self.runtime.active_catch_exception().unwrap().identifier,
                "RunMat:first"
            );
            self.polls.set(self.polls.get() + 1);
            Poll::Pending
        }
    }

    impl Drop for CheckPending {
        fn drop(&mut self) {
            assert_eq!(
                self.runtime.active_catch_exception().unwrap().identifier,
                "RunMat:first"
            );
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn interleaved_poll_and_drop_never_leak_catch_authority() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let exception = MException::new("RunMat:first".into(), "first".into());
        let polls = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let mut pending = Box::pin(first.scope_active_catch(
            Some(exception.clone()),
            CheckPending {
                runtime: first.clone(),
                polls: polls.clone(),
                drops: drops.clone(),
            },
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(pending.as_mut().poll(&mut context).is_pending());
        assert_eq!(polls.get(), 1);
        assert!(first.active_catch_exception().is_none());
        assert!(second.active_catch_exception().is_none());
        {
            let _guard = first.enter_active_catch(Some(exception));
            assert_eq!(
                first.active_catch_exception().unwrap().identifier,
                "RunMat:first"
            );
            assert!(second.active_catch_exception().is_none());
            let _mask = first.enter_active_catch(None);
            assert!(first.active_catch_exception().is_none());
        }
        drop(pending);
        assert_eq!(drops.get(), 1);
        assert!(first.active_catch_exception().is_none());
    }

    #[test]
    fn an_out_of_order_guard_removes_only_its_own_frame() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let outer = runtime
            .enter_active_catch(Some(MException::new("RunMat:outer".into(), "outer".into())));
        let inner = runtime
            .enter_active_catch(Some(MException::new("RunMat:inner".into(), "inner".into())));
        drop(outer);
        assert_eq!(
            runtime.active_catch_exception().unwrap().identifier,
            "RunMat:inner"
        );
        drop(inner);
        assert!(runtime.active_catch_exception().is_none());
    }
}
