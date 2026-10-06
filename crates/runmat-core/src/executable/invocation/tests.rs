use super::*;
use crate::{ExecutableSource, RunError, RunMatSession};

async fn infinite(session: &mut RunMatSession) -> crate::ExecutableUnit {
    session
        .compile_executable_unit(
            ExecutableSource::new("live-control", "loop.m", "while true; a = 1; end"),
            None,
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn caller_cancellation_stops_running_original_script_and_scope_is_restored() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = infinite(&mut session).await;
    let caller = Arc::new(AtomicBool::new(false));
    let control = InvocationControl::default().with_cancellation(caller.clone());
    let before = session.stats().total_executions;
    let cancel = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        caller.store(true, Ordering::Release);
    });
    let error = session
        .invoke_executable(&unit, ProcedureInvocation::entrypoint(), &control)
        .await
        .unwrap_err();
    cancel.join().unwrap();
    assert_eq!(
        session.stats().total_executions,
        before + 1,
        "must enter original execution before cancellation"
    );
    assert!(
        matches!(error, RunError::Runtime(ref error) if error.identifier() == Some("RunMat:ExecutionCancelled"))
    );
    let next = session
        .compile_executable_unit(
            ExecutableSource::new("after-control", "next.m", "ans = 42;"),
            None,
        )
        .await
        .unwrap();
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Entrypoint,
        arguments: vec![],
        requested_outputs: 1,
    };
    assert_eq!(
        session
            .invoke_executable(&next, invocation, &InvocationControl::default())
            .await
            .unwrap(),
        Value::Num(42.0)
    );
}

#[tokio::test]
async fn deadline_stops_running_original_script_with_distinct_reason() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = infinite(&mut session).await;
    let deadline = u64::try_from(runmat_time::unix_timestamp_ms()).unwrap() + 50;
    let before = session.stats().total_executions;
    let control = InvocationControl::default().with_deadline_unix_ms(deadline);
    let error = session
        .invoke_executable(&unit, ProcedureInvocation::entrypoint(), &control)
        .await
        .unwrap_err();
    assert_eq!(session.stats().total_executions, before + 1);
    assert!(
        matches!(error, RunError::Runtime(ref error) if error.identifier() == Some("RunMat:ExecutionDeadline"))
    );
}

#[tokio::test]
async fn entering_another_session_keeps_original_parent_deadline() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new("child-control", "child.m", "ans = 42;"),
            None,
        )
        .await
        .unwrap();
    let parent = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(runmat_execution::CancellationToken::default().with_deadline(Some(0)));
    let _parent = parent.enter();
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Entrypoint,
        arguments: vec![],
        requested_outputs: 1,
    };
    let result = session
        .invoke_executable(&unit, invocation, &InvocationControl::default())
        .await;
    assert!(
        matches!(result, Err(RunError::Runtime(ref error)) if error.identifier() == Some("RunMat:ExecutionDeadline")),
        "child discarded original parent deadline: {result:?}"
    );
}

async fn prepared_native_loop(session: &mut RunMatSession) -> crate::ExecutableUnit {
    let unit = session
        .compile_executable_unit(
            ExecutableSource::new(
                "native-live-control",
                "liveNative.m",
                "function y = liveNative(loop); while loop; a = 1; end; y = 42; end",
            ),
            None,
        )
        .await
        .unwrap();
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Function("liveNative".into()),
        arguments: vec![Value::Bool(false)],
        requested_outputs: 1,
    };
    assert_eq!(
        session
            .invoke_executable(
                &unit,
                invocation,
                &InvocationControl::default().force_generic_native()
            )
            .await
            .unwrap(),
        Value::Num(42.0)
    );
    assert_eq!(session.generic_native_cache_counts(), (1, 1));
    unit
}

#[tokio::test]
async fn original_caller_cancellation_stops_compiled_native_loop() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = prepared_native_loop(&mut session).await;
    let caller = Arc::new(AtomicBool::new(false));
    let control = InvocationControl::default()
        .force_generic_native()
        .with_cancellation(caller.clone());
    let cancel = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        caller.store(true, Ordering::Release);
    });
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Function("liveNative".into()),
        arguments: vec![Value::Bool(true)],
        requested_outputs: 1,
    };
    let result = session.invoke_executable(&unit, invocation, &control).await;
    cancel.join().unwrap();
    assert!(
        matches!(result, Err(RunError::Runtime(ref error)) if error.identifier() == Some("RunMat:ExecutionCancelled"))
    );
    assert_eq!(session.generic_native_cache_counts(), (1, 1));
}

#[tokio::test]
async fn deadline_stops_compiled_native_loop_with_original_reason() {
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let unit = prepared_native_loop(&mut session).await;
    let control = InvocationControl::default()
        .force_generic_native()
        .with_deadline_unix_ms(u64::try_from(runmat_time::unix_timestamp_ms()).unwrap() + 50);
    let invocation = ProcedureInvocation {
        target: ProcedureTarget::Function("liveNative".into()),
        arguments: vec![Value::Bool(true)],
        requested_outputs: 1,
    };
    let result = session.invoke_executable(&unit, invocation, &control).await;
    assert!(
        matches!(result, Err(RunError::Runtime(ref error)) if error.identifier() == Some("RunMat:ExecutionDeadline"))
    );
    assert_eq!(session.generic_native_cache_counts(), (1, 1));
}
