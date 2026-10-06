use super::*;
use runmat_execution::{CancellationReason, FutureHandle, TaskHandle};
use runmat_runtime::execution::ExecutionServiceError;

pub(super) fn reject_altered_future(service: &NativeExecutionService, original: &FutureHandle) {
    let mut altered = original.clone();
    altered.outputs.requested_outputs += 1;
    assert!(service.spawn(&altered).is_err());
    assert!(service.begin_await(Value::Future(altered.clone())).is_err());
    assert!(service
        .cancel(&Value::Future(altered.clone()), CancellationReason::User)
        .is_err());
    assert!(service
        .complete_future(&altered, Err(ExecutionServiceError::Cancelled))
        .is_err());
}

pub(super) fn reject_altered_task(service: &NativeExecutionService, original: &TaskHandle) {
    for change_generation in [false, true] {
        let mut altered = original.clone();
        if change_generation {
            altered.generation += 1;
        } else {
            altered.outputs.requested_outputs += 1;
        }
        assert!(service.begin_await(Value::Task(altered.clone())).is_err());
        assert!(service
            .cancel(&Value::Task(altered), CancellationReason::User)
            .is_err());
    }
}

pub(super) async fn verify_caller(service: &NativeExecutionService) {
    let future = service.create_future(call(2)).unwrap();
    let mut unknown = future.clone();
    unknown.id = runmat_execution::FutureId::derive(&[b"unknown-completion"]);
    assert_eq!(
        service.complete_future(&unknown, Err(ExecutionServiceError::Cancelled)),
        Err(ExecutionServiceError::UnknownHandle)
    );
    let mut foreign = future.clone();
    foreign.scope_id = runmat_execution::ExecutionScopeId::derive(&[b"foreign-completion"]);
    assert_eq!(
        service.complete_future(&foreign, Err(ExecutionServiceError::Cancelled)),
        Err(ExecutionServiceError::ForeignScope)
    );
    assert!(service
        .complete_future(&future, Err(ExecutionServiceError::Cancelled))
        .is_err());
    reject_altered_future(service, &future);
    let AwaitAction::ExecuteFuture { handle, call } =
        service.begin_await(Value::Future(future.clone())).unwrap()
    else {
        panic!("original caller did not enter execution");
    };
    assert!(service.begin_await(Value::Future(future.clone())).is_err());
    let registry: FunctionRegistry =
        serde_json::from_slice(call.program.as_ref().unwrap()).unwrap();
    let result = runmat_vm::invoke_semantic_function_value(
        call.function,
        &call.arguments,
        call.requested_outputs,
        &registry,
    )
    .await
    .unwrap();
    assert_eq!(
        service.complete_future(&handle, Ok(Value::Num(0.0))),
        Err(ExecutionServiceError::InvalidOutputContract)
    );
    service
        .complete_future(&handle, Ok(result.clone()))
        .unwrap();
    assert!(service
        .complete_future(&handle, Ok(result.clone()))
        .is_err());
    service
        .cancel(&Value::Future(handle.clone()), CancellationReason::User)
        .unwrap();
    let AwaitAction::Completed(retained) = service.begin_await(Value::Future(handle)).unwrap()
    else {
        panic!("original caller completion lost");
    };
    assert_eq!(retained, result);
    let cancelled = service.create_future(super::call(0)).unwrap();
    service
        .cancel(&Value::Future(cancelled.clone()), CancellationReason::User)
        .unwrap();
    assert_eq!(
        service.complete_future(
            &cancelled,
            Err(ExecutionServiceError::Failed("late caller".into()))
        ),
        Err(ExecutionServiceError::Cancelled)
    );
    assert!(matches!(
        service.begin_await(Value::Future(cancelled)),
        Err(ExecutionServiceError::Cancelled)
    ));
}
