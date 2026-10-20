use crate::context::RuntimeServiceFuture;
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use runmat_value::Value;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::rc::Rc;

pub(crate) struct CodecTurnService {
    pub(crate) turns: Rc<Cell<usize>>,
    pub(crate) dropped: Rc<Cell<bool>>,
    pub(crate) pending_on: usize,
}

struct PendingTurn(Rc<Cell<bool>>);

impl std::future::Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        std::task::Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

impl RuntimeExecutionServices for CodecTurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("positive quantum"))
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let turn = self.turns.get() + 1;
        self.turns.set(turn);
        if turn == self.pending_on {
            Box::pin(PendingTurn(self.dropped.clone()))
        } else {
            Box::pin(async { Ok(()) })
        }
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"net-codec-turn-test"])
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
