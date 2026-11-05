//! Explicit propagation of one run's ledger, without copying its grant or data.
use crate::UsageLedger;
use std::{cell::RefCell, future::Future, marker::PhantomData, rc::Rc, sync::Arc};

thread_local! {
    static CURRENT: RefCell<Option<UsageContext>> = const { RefCell::new(None) };
}

/// Captured by an admitted numerical owner and installed in its execution scope.
/// Clones share the same ledger; they do not mint independent operation grants.
/// This context has no account authorization or lease-renewal authority.
#[derive(Clone)]
pub struct UsageContext {
    ledger: Arc<UsageLedger>,
}

impl UsageContext {
    pub fn new(ledger: Arc<UsageLedger>) -> Self {
        Self { ledger }
    }

    /// Capture for a worker before dispatch. The worker must install its copy;
    /// thread creation alone does not propagate numerical execution authority.
    pub fn capture() -> Option<Self> {
        CURRENT.with(|slot| slot.borrow().clone())
    }

    pub fn ledger(&self) -> &UsageLedger {
        &self.ledger
    }

    pub fn delegate(
        &self,
        maximum: crate::OperationCount,
    ) -> Result<crate::DelegatedReservation, crate::LedgerError> {
        self.ledger.delegate(maximum)
    }

    pub fn enter(&self) -> UsageContextGuard {
        UsageContextGuard {
            previous: CURRENT.with(|slot| slot.replace(Some(self.clone()))),
            thread_affinity: PhantomData,
        }
    }

    pub fn run_sync<T>(&self, work: impl FnOnce() -> T) -> T {
        let _guard = self.enter();
        work()
    }

    /// Install only while polling this invocation. Pending futures leave their
    /// executor's previous context intact. Pinning introduces no heap allocation.
    pub async fn scope<F: Future>(&self, work: F) -> F::Output {
        let mut work = std::pin::pin!(work);
        std::future::poll_fn(|cx| {
            let _guard = self.enter();
            work.as_mut().poll(cx)
        })
        .await
    }
}

/// Restore on the installing thread, including on unwinding. Never move a live
/// guard to a worker: move a captured UsageContext and enter it there instead.
///
/// ```compile_fail
/// use runmat_simulation_usage::UsageContext;
/// fn move_guard(context: &UsageContext) {
///     let guard = context.enter();
///     std::thread::spawn(move || drop(guard));
/// }
/// ```
pub struct UsageContextGuard {
    previous: Option<UsageContext>,
    thread_affinity: PhantomData<Rc<()>>,
}

impl Drop for UsageContextGuard {
    fn drop(&mut self) {
        CURRENT.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OperationCount, UsageCategory};
    use std::task::{Context, Poll, Waker};

    fn context() -> UsageContext {
        let cap = OperationCount::from_operations(10);
        UsageContext::new(Arc::new(UsageLedger::new(cap, cap).unwrap()))
    }

    fn assert_current(expected: &UsageContext) {
        assert!(std::ptr::eq(
            UsageContext::capture().unwrap().ledger(),
            expected.ledger()
        ));
    }

    #[test]
    fn nested_and_worker_scopes_share_one_grant_and_restore_the_caller() {
        let outer = context();
        let inner = context();
        assert!(UsageContext::capture().is_none());
        outer.run_sync(|| {
            assert_current(&outer);
            inner.run_sync(|| assert_current(&inner));
            assert_current(&outer);
            let captured = UsageContext::capture().unwrap();
            std::thread::spawn(move || {
                assert!(UsageContext::capture().is_none());
                captured.run_sync(|| {
                    assert_current(&captured);
                    let mut reservation = captured
                        .ledger()
                        .reserve(UsageCategory::Assembly, OperationCount::from_operations(3))
                        .unwrap();
                    reservation
                        .complete(OperationCount::from_operations(3))
                        .unwrap();
                });
                assert!(UsageContext::capture().is_none());
            })
            .join()
            .unwrap();
            assert_current(&outer);
        });
        assert!(UsageContext::capture().is_none());
        assert_eq!(
            outer
                .ledger()
                .snapshot()
                .unwrap()
                .completed
                .assembly
                .operations(),
            3
        );
        assert_eq!(
            inner
                .ledger()
                .snapshot()
                .unwrap()
                .completed
                .total()
                .unwrap(),
            OperationCount::ZERO
        );
    }

    async fn work(context: &UsageContext, operations: u64) {
        assert_current(context);
        let operations = OperationCount::from_operations(operations);
        let captured = UsageContext::capture().unwrap();
        let mut reservation = captured
            .ledger()
            .reserve(UsageCategory::Solve, operations)
            .unwrap();
        let mut yielded = false;
        std::future::poll_fn(|cx| {
            assert_current(context);
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
        .await;
        assert_current(context);
        reservation.complete(operations).unwrap();
    }

    #[test]
    fn interleaved_and_cancelled_futures_do_not_leak_execution_context_or_reservations() {
        let first = context();
        let second = context();
        let mut first_work = Box::pin(first.scope(work(&first, 2)));
        let mut second_work = Box::pin(second.scope(work(&second, 3)));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(first_work.as_mut().poll(&mut cx).is_pending());
        assert!(UsageContext::capture().is_none());
        assert!(second_work.as_mut().poll(&mut cx).is_pending());
        assert!(UsageContext::capture().is_none());
        assert!(first_work.as_mut().poll(&mut cx).is_ready());
        assert!(UsageContext::capture().is_none());
        drop(second_work);
        assert!(UsageContext::capture().is_none());
        assert_eq!(first.ledger().finish().unwrap().solve.operations(), 2);
        assert_eq!(
            second.ledger().snapshot().unwrap().reserved,
            OperationCount::ZERO
        );
        assert_eq!(
            second.ledger().finish().unwrap().total().unwrap(),
            OperationCount::ZERO
        );
    }
}
