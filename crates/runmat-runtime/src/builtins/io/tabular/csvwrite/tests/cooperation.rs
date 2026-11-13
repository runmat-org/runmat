use super::*;
use crate::context::{RuntimeContext, RuntimeServiceFuture};
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Context, Poll};

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

struct PendingWriterService {
    turns: Rc<Cell<usize>>,
    dropped: Rc<Cell<bool>>,
}

impl RuntimeExecutionServices for PendingWriterService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum"))
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        self.turns.set(self.turns.get() + 1);
        Box::pin(PendingTurn(self.dropped.clone()))
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"tabular-writer-test"])
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn actual_csvwrite_future_abandons_during_writer_without_publication() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/tabular",
    ));
    let _guard = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir("/tabular").expect("cwd");

    let turns = Rc::new(Cell::new(0));
    let dropped = Rc::new(Cell::new(false));
    let service = Rc::new(PendingWriterService {
        turns: turns.clone(),
        dropped: dropped.clone(),
    });
    let source = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(service, source.token());
    let mut call = Box::pin(super::super::csvwrite_builtin(
        Some(runtime),
        Value::from("partial.csv"),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(turns.get(), 1, "first service turn is the writer row");
    drop(call);
    assert!(dropped.get(), "abandonment drops the pending service turn");

    let returned = futures::executor::block_on(super::super::csvwrite_builtin(
        None,
        Value::from("partial.csv"),
        Value::Num(9.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("subsequent write retains selected owner");
    assert!(matches!(returned, Value::Num(2.0)));
    assert_eq!(
        provider
            .read_project_path("/tabular/partial.csv")
            .expect("replacement output"),
        b"9\n"
    );
}
