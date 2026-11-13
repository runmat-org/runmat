use super::*;
use runmat_execution::{CancellationReason, FutureHandle, TaskHandle};
use runmat_runtime::execution::ExecutionServiceError;

pub(super) fn reject_altered_future(service: &NativeExecutionService, original: &FutureHandle) {
    let mut altered = original.clone();
    altered.outputs.requested_outputs += 1;
    assert!(service.spawn(&altered).is_err());
    assert!(service
        .begin_await(Value::Future(altered.clone().into()))
        .is_err());
    assert!(service
        .cancel(
            &Value::Future(altered.clone().into()),
            CancellationReason::User
        )
        .is_err());
    assert!(service
        .complete_future((&altered).into(), Err(ExecutionServiceError::Cancelled))
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
        assert!(service
            .begin_await(Value::Task(altered.clone().into()))
            .is_err());
        assert!(service
            .cancel(&Value::Task(altered.into()), CancellationReason::User)
            .is_err());
    }
}

pub(super) async fn verify_caller(service: &NativeExecutionService) {
    let original = call(2).await;
    let future = service.create_future(original.call.clone()).unwrap();
    let mut unknown = future.clone();
    unknown.id = runmat_execution::FutureId::derive(&[b"unknown-completion"]);
    assert_eq!(
        service.complete_future((&unknown).into(), Err(ExecutionServiceError::Cancelled)),
        Err(ExecutionServiceError::UnknownHandle)
    );
    let mut foreign = future.clone();
    foreign.scope_id = runmat_execution::ExecutionScopeId::derive(&[b"foreign-completion"]);
    assert_eq!(
        service.complete_future((&foreign).into(), Err(ExecutionServiceError::Cancelled)),
        Err(ExecutionServiceError::ForeignScope)
    );
    assert!(service
        .complete_future((&future).into(), Err(ExecutionServiceError::Cancelled))
        .is_err());
    reject_altered_future(service, &future);
    let AwaitAction::ExecuteFuture { handle, call } = service
        .begin_await(Value::Future(future.clone().into()))
        .unwrap()
    else {
        panic!("original caller did not enter execution");
    };
    assert!(service
        .begin_await(Value::Future(future.clone().into()))
        .is_err());
    assert!(matches!(
        &call.program,
        runmat_runtime::execution::DeferredProgram::SelectedGraph(_)
    ));
    assert!(call.invocation.arguments().is_empty());
    let requested_outputs = call.invocation.requested_outputs();
    let mut caller = RunMatSession::with_options(false, false).unwrap();
    let result = caller
        .invoke_portable_unit(
            &original.envelope,
            vec![],
            requested_outputs,
            &runmat_core::InvocationControl::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        service.complete_future(
            (&handle).into(),
            Ok(runmat_runtime::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(0.0)).unwrap(),
            )
            .unwrap())
        ),
        Err(ExecutionServiceError::InvalidOutputContract)
    );
    service
        .complete_future((&handle).into(), Ok(result.clone()))
        .unwrap();
    assert!(service
        .complete_future((&handle).into(), Ok(result.clone()))
        .is_err());
    service
        .cancel(&handle.retained_value(), CancellationReason::User)
        .unwrap();
    let AwaitAction::Completed(retained) = service.begin_await(handle.retained_value()).unwrap()
    else {
        panic!("original caller completion lost");
    };
    assert_eq!(retained, result);
    let cancelled_original = super::call(0).await;
    let cancelled = service
        .create_future(cancelled_original.call.clone())
        .unwrap();
    service
        .cancel(
            &Value::Future(cancelled.clone().into()),
            CancellationReason::User,
        )
        .unwrap();
    assert_eq!(
        service.complete_future(
            (&cancelled).into(),
            Err(ExecutionServiceError::Failed("late caller".into()))
        ),
        Err(ExecutionServiceError::Cancelled)
    );
    assert!(matches!(
        service.begin_await(Value::Future(cancelled.into())),
        Err(ExecutionServiceError::Cancelled)
    ));
}
