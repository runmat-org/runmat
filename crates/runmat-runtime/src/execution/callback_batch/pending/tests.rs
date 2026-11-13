//! Compact accepted-task cleanup controls using the existing serial state
//! owner. Only batch-retirement observation is added; no worker framework.
use super::*;
use crate::execution::{
    DeferredProgram, FutureCompletionRef, RuntimeExecutionService, RuntimeExecutionServices,
};
use runmat_execution::{
    ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle, PoolRequest, PoolSnapshot,
    TaskResultClaim,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Service {
    inner: RuntimeExecutionService,
    cancelled: RefCell<Vec<TaskHandle>>,
    released: RefCell<Vec<TaskHandle>>,
}
impl RuntimeExecutionServices for Service {
    fn scope_id(&self) -> ExecutionScopeId {
        self.inner.scope_id()
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        self.inner.current_pool()
    }
    fn ensure_pool(&self, request: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        self.inner.ensure_pool(request)
    }
    fn close_pool(&self, pool: &PoolHandle) -> Result<(), ExecutionServiceError> {
        self.inner.close_pool(pool)
    }
    fn create_future(&self, call: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        self.inner.create_future(call)
    }
    fn spawn(&self, future: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        self.inner.spawn(future)
    }
    fn inspect_future(
        &self,
        future: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.inner.inspect_future(future)
    }
    fn inspect_task(
        &self,
        task: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        self.inner.inspect_task(task)
    }
    fn claim_next_task_result(
        &self,
        tasks: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        self.inner.claim_next_task_result(tasks)
    }
    fn mark_task_result_read(&self, task: &TaskHandle) -> Result<(), ExecutionServiceError> {
        self.inner.mark_task_result_read(task)
    }
    fn begin_await(&self, value: Value) -> Result<AwaitAction, ExecutionServiceError> {
        self.inner.begin_await(value)
    }
    fn complete_future(
        &self,
        future: FutureCompletionRef<'_>,
        result: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.inner.complete_future(future, result)
    }
    fn drain_scope(&self, reason: CancellationReason) -> () {
        self.inner.drain_scope(reason)
    }
    fn cancel(
        &self,
        value: &Value,
        reason: CancellationReason,
    ) -> Result<(), ExecutionServiceError> {
        if let Value::Task(task) = value {
            self.cancelled.borrow_mut().push((**task).clone());
        }
        self.inner.cancel(value, reason)
    }
    fn release_callback_batch(&self, tasks: &[TaskHandle]) -> Result<(), ExecutionServiceError> {
        self.released.borrow_mut().extend_from_slice(tasks);
        Ok(())
    }
}
fn call() -> DeferredCall {
    DeferredCall {
        spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
        invocation: DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::builtin("abs").unwrap(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![Value::Num(1.0)],
            requested_outputs: 1,
        },
        retry: RetryPolicy::Never,
        program_revision: None,
        environment: Default::default(),
        console: None,
        capabilities: runmat_builtins::builtin_required_capabilities("abs"),
        program: DeferredProgram::Local,
    }
}
fn accepted(service: &Service) -> (FutureHandle, TaskHandle) {
    let future = service.create_future(call()).unwrap();
    let task = service.spawn(&future).unwrap();
    (future, task)
}
fn complete(service: &Service, future: &FutureHandle, value: f64) {
    assert!(matches!(
        service
            .begin_await(Value::Future(future.clone().into()))
            .unwrap(),
        AwaitAction::ExecuteFuture { .. }
    ));
    service
        .complete_future(
            future.into(),
            Ok(RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(value)).unwrap(),
            )
            .unwrap()),
        )
        .unwrap();
}

#[test]
fn unpolled_accepted_owner_retires_exact_task_and_reordered_results_stay_ordered() {
    let service = Rc::new(Service::default());
    let runtime = RuntimeContext::new(service.clone());
    let original = runmat_execution::cancellation::CancellationSource::new();
    let public = service.create_future(call()).unwrap();
    let (_future, task) = accepted(&service);
    let pending = PendingBatch::new(&runtime, vec![task.clone()], original.token(), 1).unwrap();
    // This is the same immediate accepted-owner future held by start_call.
    let unpolled = wait(pending);
    drop(unpolled);
    assert_eq!(*service.cancelled.borrow(), vec![task.clone()]);
    assert_eq!(*service.released.borrow(), vec![task]);
    assert!(service.inspect_future(&public).is_ok());

    service.cancelled.borrow_mut().clear();
    service.released.borrow_mut().clear();
    let (first, first_task) = accepted(&service);
    let (second, second_task) = accepted(&service);
    complete(&service, &second, 22.0);
    complete(&service, &first, 11.0);
    let tasks = vec![first_task, second_task];
    let pending = PendingBatch::new(&runtime, tasks.clone(), original.token(), 2).unwrap();
    let returned = futures::executor::block_on(wait(pending)).unwrap();
    let values: Vec<_> = returned
        .into_iter()
        .map(|v| v.try_into_single().unwrap().value().clone())
        .collect();
    assert_eq!(values, vec![Value::Num(11.0), Value::Num(22.0)]);
    assert!(service.cancelled.borrow().is_empty());
    assert_eq!(*service.released.borrow(), tasks);
    assert!(service.inspect_future(&public).is_ok());
}
