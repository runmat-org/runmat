//! Invocation-local CPU work cooperation through the owning execution service.

use std::num::NonZeroU64;
use std::rc::Rc;

use futures::future::{select, Either};
use runmat_execution::cancellation::CancellationToken;

use crate::context::RuntimeServiceFuture;

use super::{ExecutionServiceError, RuntimeExecutionServices};

#[path = "cooperation/cancellable_await.rs"]
mod cancellable_await;

#[cfg(test)]
type CpuTurn = dyn Fn() -> RuntimeServiceFuture<Result<(), ExecutionServiceError>>;

enum CpuTurnOwner {
    Service(Rc<dyn RuntimeExecutionServices>),
    #[cfg(test)]
    Callback(Rc<CpuTurn>),
}

impl CpuTurnOwner {
    fn start(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        match self {
            Self::Service(service) => service.cooperate_cpu(),
            #[cfg(test)]
            Self::Callback(turn) => turn(),
        }
    }
}

/// The execution adapter, not an individual builtin, chooses whether CPU work
/// must periodically give other tasks a turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuCooperationPolicy {
    NeverYield,
    WorkQuantum(NonZeroU64),
}

pub struct CpuCooperationBudget {
    turn: Option<CpuTurnOwner>,
    cancellation: CancellationToken,
    policy: CpuCooperationPolicy,
    pending_work: u64,
}

impl CpuCooperationBudget {
    /// The host-composed native driver retains `retained` until an indivisible
    /// CPU kernel physically completes. Cancellation stops this await, not
    /// the library kernel or its managed reservation.
    pub async fn run_owned_blocking<T, G, F>(
        &self,
        service: Option<&dyn super::blocking_cpu::RuntimeBlockingCpuService>,
        retained: G,
        work: F,
    ) -> Result<T, ExecutionServiceError>
    where
        T: Send + 'static,
        G: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let service = service.ok_or_else(|| {
                ExecutionServiceError::Infrastructure(
                    "owned blocking CPU service was not installed by this host".into(),
                )
            })?;
            if self.cancellation.is_cancelled() {
                return Err(ExecutionServiceError::Cancelled);
            }
            let (send, receive) = futures::channel::oneshot::channel();
            let cancellation = self.cancellation.clone();
            let job = Box::new(move || {
                let _retained = retained;
                let result = if cancellation.is_cancelled() {
                    Err(ExecutionServiceError::Cancelled)
                } else {
                    Ok(work())
                };
                let _ = send.send(result);
            });
            self.await_cancellable(service.submit(job)).await??;
            return self.await_cancellable(receive).await?.map_err(|_| {
                ExecutionServiceError::WorkerLost(
                    "owned blocking CPU job ended before publication".into(),
                )
            })?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Portable same-thread math remains available, but this is not
            // a claim of mid-kernel yielding until browser worker transport
            // can retain the same invocation resource owner.
            let _ = service;
            if self.cancellation.is_cancelled() {
                return Err(ExecutionServiceError::Cancelled);
            }
            let _retained = retained;
            let result = work();
            if self.cancellation.is_cancelled() {
                return Err(ExecutionServiceError::Cancelled);
            }
            Ok(result)
        }
    }

    /// Await an invocation-owned operation, dropping it when cancellation
    /// wins. This is an execution boundary, not a second scheduling policy.
    pub async fn await_cancellable<F: std::future::Future>(
        &self,
        future: F,
    ) -> Result<F::Output, ExecutionServiceError> {
        cancellable_await::run(&self.cancellation, future).await
    }

    pub(crate) fn new(
        service: Rc<dyn RuntimeExecutionServices>,
        cancellation: CancellationToken,
    ) -> Self {
        let policy = service.cpu_cooperation_policy();
        let turn = match policy {
            CpuCooperationPolicy::NeverYield => None,
            // Retain the already-owned adapter directly; a budget needs no
            // separately allocated closure to call this same service.
            CpuCooperationPolicy::WorkQuantum(_) => Some(CpuTurnOwner::Service(service)),
        };
        Self {
            turn,
            cancellation,
            policy,
            pending_work: 0,
        }
    }

    /// Charge a completed unit of CPU work and yield only when the adapter's
    /// quantum is exhausted. Native `NeverYield` does not allocate a future.
    pub async fn checkpoint(&mut self, completed_work: u64) -> Result<(), ExecutionServiceError> {
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        let CpuCooperationPolicy::WorkQuantum(quantum) = self.policy else {
            return Ok(());
        };
        let total = u128::from(self.pending_work) + u128::from(completed_work);
        if total < u128::from(quantum.get()) {
            self.pending_work = total as u64;
            return Ok(());
        }
        self.pending_work = (total % u128::from(quantum.get())) as u64;
        let turn = self.turn.as_ref().expect("a work quantum owns a turn hook");
        let result = match select(turn.start(), self.cancellation.cancelled()).await {
            Either::Left((result, _listener)) => result,
            Either::Right((_reason, _turn)) => return Err(ExecutionServiceError::Cancelled),
        };
        if self.cancellation.is_cancelled() {
            return Err(ExecutionServiceError::Cancelled);
        }
        result?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn with_test_turn(
        policy: CpuCooperationPolicy,
        cancellation: CancellationToken,
        turn: impl Fn() -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> + 'static,
    ) -> Self {
        Self {
            turn: Some(CpuTurnOwner::Callback(Rc::new(turn))),
            cancellation,
            policy,
            pending_work: 0,
        }
    }
}

#[cfg(test)]
#[path = "cooperation/service_tests.rs"]
mod service_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    #[test]
    fn native_policy_never_calls_turn_and_quantum_accounts_exactly() {
        let source = CancellationSource::new();
        let serial = Rc::new(super::super::RuntimeExecutionService::new());
        let mut actual_native = CpuCooperationBudget::new(serial, source.token());
        assert!(actual_native.turn.is_none());
        futures::executor::block_on(actual_native.checkpoint(u64::MAX)).unwrap();
        let calls = Rc::new(Cell::new(0));
        let count = calls.clone();
        let mut never = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::NeverYield,
            source.token(),
            move || {
                count.set(count.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        futures::executor::block_on(never.checkpoint(u64::MAX)).unwrap();
        assert_eq!(calls.get(), 0);

        let count = calls.clone();
        let mut quantum = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(7).unwrap()),
            source.token(),
            move || {
                count.set(count.get() + 1);
                Box::pin(async { Ok(()) })
            },
        );
        futures::executor::block_on(quantum.checkpoint(6)).unwrap();
        assert_eq!(calls.get(), 0);
        futures::executor::block_on(quantum.checkpoint(1)).unwrap();
        assert_eq!(calls.get(), 1);
        futures::executor::block_on(quantum.checkpoint(u64::MAX)).unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(quantum.pending_work, (u128::from(u64::MAX) % 7) as u64);
        futures::executor::block_on(quantum.checkpoint(2)).unwrap();
        let pending = quantum.pending_work;
        futures::executor::block_on(quantum.checkpoint(u64::MAX)).unwrap();
        assert_eq!(calls.get(), 3);
        assert_eq!(
            quantum.pending_work,
            ((u128::from(pending) + u128::from(u64::MAX)) % 7) as u64
        );
    }

    #[test]
    fn pending_turn_is_cancelled_and_dropped_with_its_listener() {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        struct PendingTurn(Rc<Cell<bool>>);
        impl Future for PendingTurn {
            type Output = Result<(), ExecutionServiceError>;
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        impl Drop for PendingTurn {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        let dropped_turn = dropped.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(dropped_turn.clone())),
        );
        let mut future = Box::pin(budget.checkpoint(1));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        source.cancel(CancellationReason::User);
        assert!(matches!(
            future.as_mut().poll(&mut context),
            Poll::Ready(Err(ExecutionServiceError::Cancelled))
        ));
        drop(future);
        assert!(dropped.get());
    }

    #[test]
    fn turn_failure_and_preexisting_cancel_are_not_success() {
        let source = CancellationSource::new();
        let mut failing = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("lost turn".into())) }),
        );
        assert!(matches!(
            futures::executor::block_on(failing.checkpoint(1)),
            Err(ExecutionServiceError::Infrastructure(_))
        ));
        source.cancel(CancellationReason::User);
        assert!(matches!(
            futures::executor::block_on(failing.checkpoint(1)),
            Err(ExecutionServiceError::Cancelled)
        ));

        let after = CancellationSource::new();
        let cancel_during_turn = after.clone();
        let mut budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            after.token(),
            move || {
                cancel_during_turn.cancel(CancellationReason::User);
                Box::pin(async { Ok(()) })
            },
        );
        assert!(matches!(
            futures::executor::block_on(budget.checkpoint(1)),
            Err(ExecutionServiceError::Cancelled)
        ));
    }
}
