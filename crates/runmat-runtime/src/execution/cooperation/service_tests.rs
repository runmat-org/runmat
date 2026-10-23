use super::*;
use crate::context::RuntimeContext;
use crate::execution::RootedValueSequence;
use crate::execution::{AwaitAction, DeferredCall};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::Value;
use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Clone)]
enum TurnBehavior {
    Ready,
    Pending(Rc<Cell<bool>>),
    Failure(CancellationSource),
}

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

struct TestTurnService {
    policy: CpuCooperationPolicy,
    calls: Rc<Cell<u32>>,
    behavior: TurnBehavior,
}

impl RuntimeExecutionServices for TestTurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        self.policy
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        self.calls.set(self.calls.get() + 1);
        match &self.behavior {
            TurnBehavior::Ready => Box::pin(async { Ok(()) }),
            TurnBehavior::Pending(dropped) => Box::pin(PendingTurn(dropped.clone())),
            TurnBehavior::Failure(cancel) => {
                cancel.cancel(CancellationReason::User);
                Box::pin(async { Err(ExecutionServiceError::Infrastructure("turn failed".into())) })
            }
        }
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"cpu-cooperation-test"])
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        Ok(None)
    }
    fn ensure_pool(&self, _: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn close_pool(&self, _: &PoolHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn create_future(&self, _: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn spawn(&self, _: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_future(
        &self,
        _: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_task(
        &self,
        _: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn claim_next_task_result(
        &self,
        _: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn mark_task_result_read(&self, _: &TaskHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn begin_await(&self, _: Value) -> Result<AwaitAction, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn complete_future(
        &self,
        _: crate::execution::FutureCompletionRef<'_>,
        _: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn cancel(&self, _: &Value, _: CancellationReason) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn drain_scope(&self, _: CancellationReason) {}
}

#[test]
fn context_uses_actual_adapter_policy_and_turn_hook() {
    let source = CancellationSource::new();
    let calls = Rc::new(Cell::new(0));
    let service = Rc::new(TestTurnService {
        policy: CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(7).unwrap()),
        calls: calls.clone(),
        behavior: TurnBehavior::Ready,
    });
    let authority: Rc<dyn RuntimeExecutionServices> = service.clone();
    let context = RuntimeContext::with_cancellation(authority.clone(), source.token());
    let mut budget = context.cpu_cooperation();
    assert!(
        matches!(&budget.turn, Some(CpuTurnOwner::Service(retained)) if Rc::ptr_eq(retained, &authority))
    );
    futures::executor::block_on(budget.checkpoint(6)).unwrap();
    assert_eq!(calls.get(), 0);
    futures::executor::block_on(budget.checkpoint(1)).unwrap();
    assert_eq!(calls.get(), 1);

    let fresh = CancellationSource::new();
    let rebound = context.clone().with_cancellation_token(fresh.token());
    fresh.cancel(CancellationReason::User);
    let mut rebound_budget = rebound.cpu_cooperation();
    assert!(matches!(
        futures::executor::block_on(rebound_budget.checkpoint(7)),
        Err(ExecutionServiceError::Cancelled)
    ));
    assert_eq!(calls.get(), 1, "cancelled generation must not call service");
    futures::executor::block_on(budget.checkpoint(7)).unwrap();
    assert_eq!(calls.get(), 2, "old invocation retains its original token");
}

#[test]
fn abandoning_pending_context_turn_releases_the_service_future() {
    let source = CancellationSource::new();
    let dropped = Rc::new(Cell::new(false));
    let service = Rc::new(TestTurnService {
        policy: CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        calls: Rc::new(Cell::new(0)),
        behavior: TurnBehavior::Pending(dropped.clone()),
    });
    let mut budget = CpuCooperationBudget::new(service, source.token());
    let mut pending = Box::pin(budget.checkpoint(1));
    let waker = futures::task::noop_waker();
    assert!(pending
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(pending);
    assert!(dropped.get());
    assert!(!source.token().is_cancelled());
}

#[test]
fn cancellation_wins_when_service_reports_failure_in_same_turn() {
    let source = CancellationSource::new();
    let service = Rc::new(TestTurnService {
        policy: CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        calls: Rc::new(Cell::new(0)),
        behavior: TurnBehavior::Failure(source.clone()),
    });
    let mut budget = CpuCooperationBudget::new(service, source.token());
    assert!(matches!(
        futures::executor::block_on(budget.checkpoint(1)),
        Err(ExecutionServiceError::Cancelled)
    ));
}
