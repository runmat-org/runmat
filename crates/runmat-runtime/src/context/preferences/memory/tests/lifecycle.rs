use super::*;
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::{CellArray, HandleRef};
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::task::{Context, Poll};

/// The second charged turn occurs after the candidate contains its first entry,
/// while the first transaction still owns the document mutex.
struct StagedTurnService {
    calls: Cell<u32>,
    pending_turn_dropped: Rc<Cell<bool>>,
}

impl StagedTurnService {
    fn new() -> Self {
        Self {
            calls: Cell::new(0),
            pending_turn_dropped: Rc::new(Cell::new(false)),
        }
    }
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

impl RuntimeExecutionServices for StagedTurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap())
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == 3 {
            Box::pin(PendingTurn(self.pending_turn_dropped.clone()))
        } else {
            Box::pin(async { Ok(()) })
        }
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"preference-staged-turn-test"])
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

fn poll_once<F: Future + ?Sized>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = futures::task::noop_waker();
    future.poll(&mut Context::from_waker(&waker))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_default_future_roots_caller_handle_until_drop() {
    runmat_gc::gc_test_context(|| {
        let service = IsolatedMemoryPreferenceService::new();
        block_on(service.set_many(
            runtime(),
            "app".into(),
            vec![("existing".into(), Value::Num(3.0))],
        ))
        .unwrap();
        let document = block_on(service.document.lock());
        let initial = runmat_gc::gc_allocate_rooted(Value::String("pending".into())).unwrap();
        let handle = initial.handle();
        let default = Value::Cell(
            CellArray::new(
                vec![Value::HandleObject(HandleRef {
                    class_name: "PendingPreference".into(),
                    target: handle,
                    valid: true,
                })],
                1,
                1,
            )
            .unwrap(),
        );
        let cancellation = CancellationSource::new();
        let mut future = service.read_or_create_many(
            RuntimeContext::with_cancellation(
                Rc::new(crate::execution::RuntimeExecutionService::new()),
                cancellation.token(),
            ),
            "app".into(),
            vec!["existing".into()],
            Some(vec![default]),
        );
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "unpolled future roots input"
        );
        assert!(poll_once(future.as_mut()).is_pending());
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        cancellation.cancel(CancellationReason::User);
        assert!(matches!(
            poll_once(future.as_mut()),
            Poll::Ready(Err(PreferenceServiceError::Execution(
                ExecutionServiceError::Cancelled
            )))
        ));
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
        drop(document);
        assert_eq!(
            block_on(service.read_or_create_many(
                runtime(),
                "app".into(),
                vec!["existing".into()],
                None,
            ))
            .unwrap(),
            vec![Value::Num(3.0)],
            "cancelled ignored default cannot mutate the stored value"
        );
        block_on(service.set_many(
            runtime(),
            "app".into(),
            vec![("fresh".into(), Value::Num(4.0))],
        ))
        .unwrap();
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_mutex_waiter_does_not_release_or_publish_the_first_transaction() {
    let execution = Rc::new(StagedTurnService::new());
    let service = IsolatedMemoryPreferenceService::new();
    let first_cancel = CancellationSource::new();
    let first_runtime = RuntimeContext::with_cancellation(execution.clone(), first_cancel.token());
    let mut first = Box::pin(service.set_many(
        first_runtime,
        "app".into(),
        vec![("first".into(), Value::Num(1.0))],
    ));
    assert!(poll_once(first.as_mut()).is_pending());
    assert_eq!(
        execution.calls.get(),
        3,
        "first call reached staged checkpoint"
    );

    let waiter_cancel = CancellationSource::new();
    let waiter_runtime =
        RuntimeContext::with_cancellation(execution.clone(), waiter_cancel.token());
    let mut waiter = Box::pin(service.set_many(
        waiter_runtime,
        "app".into(),
        vec![("waiter".into(), Value::Num(2.0))],
    ));
    assert!(poll_once(waiter.as_mut()).is_pending());
    assert_eq!(
        execution.calls.get(),
        5,
        "waiter passed its charged preparation turns"
    );
    waiter_cancel.cancel(CancellationReason::User);
    assert!(matches!(
        poll_once(waiter.as_mut()),
        Poll::Ready(Err(PreferenceServiceError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(waiter);
    assert!(
        !execution.pending_turn_dropped.get(),
        "waiter cannot retire first turn"
    );

    first_cancel.cancel(CancellationReason::User);
    assert!(matches!(
        poll_once(first.as_mut()),
        Poll::Ready(Err(PreferenceServiceError::Execution(
            ExecutionServiceError::Cancelled
        )))
    ));
    drop(first);
    assert!(execution.pending_turn_dropped.get());
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());
    block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("fresh".into(), Value::Num(3.0))],
    ))
    .unwrap();
    let stored = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored["fresh"], Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dropped_mutex_waiter_and_staged_owner_leave_same_service_usable() {
    let execution = Rc::new(StagedTurnService::new());
    let service = IsolatedMemoryPreferenceService::new();
    let mut first = Box::pin(service.set_many(
        RuntimeContext::new(execution.clone()),
        "app".into(),
        vec![("first".into(), Value::Num(1.0))],
    ));
    assert!(poll_once(first.as_mut()).is_pending());
    assert_eq!(execution.calls.get(), 3);

    let mut waiter = Box::pin(service.set_many(
        RuntimeContext::new(execution.clone()),
        "app".into(),
        vec![("waiter".into(), Value::Num(2.0))],
    ));
    assert!(poll_once(waiter.as_mut()).is_pending());
    assert_eq!(
        execution.calls.get(),
        5,
        "waiter is pending on the owned document"
    );
    drop(waiter);
    drop(first);
    assert!(execution.pending_turn_dropped.get());
    assert!(block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .is_none());

    block_on(service.set_many(
        runtime(),
        "app".into(),
        vec![("fresh".into(), Value::Num(4.0))],
    ))
    .unwrap();
    let stored = block_on(service.group_values(runtime(), "app".into()))
        .unwrap()
        .unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored["fresh"], Value::Num(4.0));
}
