use super::{reserve, settle_failed, settle_submission_failed, FutureState, State};
use runmat_execution::{FutureHandle, FutureId, OutputContract, RetryPolicy};
use runmat_runtime::call::descriptor::{CallableDescriptor, CallableMetadata, CallableTarget};
use runmat_runtime::execution::{
    DeferredCall, DeferredInvocation, DeferredProgram, ExecutionServiceError,
};
use runmat_value::Value;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;

#[cfg(unix)]
fn service_fixture() -> (tempfile::TempDir, super::NativeExecutionService) {
    use runmat_execution::resource::Capability;
    use std::collections::BTreeSet;

    let temporary = tempfile::tempdir().expect("temporary native session");
    let config = crate::NativeExecutionConfig {
        executable: "/bin/sh".into(),
        worker_arguments: vec!["-c".into(), "exit 0".into()],
        max_workers: 1,
        max_message_bytes: 64 * 1024,
        max_object_bytes: 64 * 1024,
        max_stderr_bytes: 4096,
        store_root: temporary.path().join("session"),
        worker_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        host_inventory: runmat_core::RunMatSession::with_options(false, false)
            .unwrap()
            .execution_host_inventory(
                runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
            )
            .unwrap(),
        accelerator_devices: Vec::new(),
    };
    (
        temporary,
        super::NativeExecutionService::new(config).expect("native execution service"),
    )
}

#[cfg(unix)]
fn admitted_program_call() -> DeferredCall {
    let program = b"unexecuted checkpoint witness".to_vec();
    let digest = runmat_execution::Digest::sha256(&program);
    DeferredCall {
        invocation: DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                None,
            ),
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: Vec::new(),
            requested_outputs: 1,
        },
        retry: RetryPolicy::Never,
        program_revision: Some(
            runmat_execution::ProgramRevision::new(
                digest,
                digest,
                runmat_core::program_environment(runmat_core::CompatMode::Matlab),
            )
            .expect("host-compatible checkpoint program"),
        ),
        capabilities: Default::default(),
        program: DeferredProgram::InterpreterV2(program.into()),
    }
}

fn fixture() -> (Mutex<State>, FutureHandle) {
    let scope_id = crate::config::fresh_scope_id(b"materialization-test", 1);
    let id = FutureId::derive(&[scope_id.bytes(), b"future"]);
    let call = DeferredCall {
        invocation: DeferredInvocation::callable(CallableDescriptor {
            target: CallableTarget::FevalForward(Value::Num(13.0)),
            args: vec![Value::Num(29.0)],
            requested_outputs: 1,
            metadata: CallableMetadata::default(),
        }),
        retry: RetryPolicy::Never,
        program_revision: None,
        capabilities: Default::default(),
        program: DeferredProgram::Local,
    };
    let state = State {
        next_future: 1,
        next_task: 0,
        next_materialization: 0,
        futures: HashMap::from([(id, FutureState::Deferred(Rc::new(call)))]),
        tasks: HashMap::new(),
        graph_jobs: HashMap::new(),
        cancelling_graphs: Vec::new(),
        pool_generation: 1,
        pool_open: true,
    };
    (
        Mutex::new(state),
        FutureHandle {
            id,
            scope_id,
            outputs: OutputContract {
                requested_outputs: 1,
            },
        },
    )
}

#[test]
fn materialization_runs_outside_lock_and_keeps_the_entire_invocation_rooted() {
    let (state, future) = fixture();
    let (reservation, call) = reserve(&state, &future).unwrap();
    assert!(state.try_lock().is_ok(), "codec may reenter the service");
    assert!(matches!(
        reserve(&state, &future),
        Err(ExecutionServiceError::Failed(_))
    ));
    let mut observed = Vec::new();
    {
        let state = state.lock().unwrap();
        let FutureState::Materializing(active) = state.futures.get(&future.id).unwrap() else {
            panic!("reserved call must stay in the root visitor");
        };
        active
            .call
            .invocation
            .visit_retained_values(&mut |value| {
                observed.push(value.clone());
                Some(())
            })
            .expect("root visitor must inspect both retained values");
    }
    assert_eq!(observed, vec![Value::Num(29.0), Value::Num(13.0)]);
    {
        let state = state.lock().unwrap();
        let retained = match state.futures.get(&future.id).unwrap() {
            FutureState::Materializing(active) => &active.call,
            _ => panic!("reservation changed"),
        };
        assert!(Rc::ptr_eq(&call, retained));
    }
    settle_failed(&state, &future, reservation).unwrap();
    assert!(matches!(
        state.lock().unwrap().futures.get(&future.id),
        Some(FutureState::Deferred(_))
    ));
}

#[test]
fn cancelled_materialization_retires_once_after_snapshot_failure() {
    let (state, future) = fixture();
    let (reservation, call) = reserve(&state, &future).unwrap();
    {
        let mut state = state.lock().unwrap();
        let FutureState::Materializing(active) = state.futures.get_mut(&future.id).unwrap() else {
            panic!("reservation missing");
        };
        active.cancelled = true;
        assert!(Rc::strong_count(&active.call) >= 2);
    }
    assert!(matches!(
        settle_failed(&state, &future, reservation),
        Err(ExecutionServiceError::Cancelled)
    ));
    assert!(matches!(
        state.lock().unwrap().futures.get(&future.id),
        Some(FutureState::Cancelled)
    ));
    assert_eq!(Rc::strong_count(&call), 1);
    assert!(matches!(
        settle_failed(&state, &future, reservation),
        Err(ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn ambiguous_driver_failure_is_terminal_and_cannot_requeue_the_call() {
    let (state, future) = fixture();
    let (reservation, call) = reserve(&state, &future).unwrap();
    let failure = ExecutionServiceError::Failed("driver checkpoint failed".into());
    settle_submission_failed(&state, &future, reservation, failure.clone()).unwrap();
    assert!(matches!(
        state.lock().unwrap().futures.get(&future.id),
        Some(FutureState::Completed(Err(error))) if *error == failure
    ));
    assert_eq!(Rc::strong_count(&call), 1);
    assert!(matches!(
        reserve(&state, &future),
        Err(ExecutionServiceError::Failed(_))
    ));
}

#[cfg(unix)]
#[test]
fn caller_completion_cannot_overwrite_a_deferred_or_accepted_future() {
    use runmat_runtime::execution::{AwaitAction, RootedValueSequence, RuntimeExecutionServices};

    let (_temporary, service) = service_fixture();
    let future = service.create_future(admitted_program_call()).unwrap();
    let result =
        RootedValueSequence::new(runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap())
            .unwrap();
    assert!(service
        .complete_future(&future, Ok(result.clone()))
        .is_err());
    assert!(matches!(
        service.begin_await(Value::Future(future.clone())).unwrap(),
        AwaitAction::ExecuteFuture { .. }
    ));
    service.complete_future(&future, Ok(result)).unwrap();
    assert!(service
        .complete_future(&future, Err(ExecutionServiceError::Cancelled))
        .is_err());
}

#[cfg(unix)]
#[test]
fn accepted_checkpoint_failure_cancels_exact_task_and_cannot_submit_twice() {
    use runmat_execution::state::TaskState;
    use runmat_execution::ExecutionHandleState;
    use runmat_execution_runner::driver::DriverEventKind;
    use runmat_runtime::execution::RuntimeExecutionServices;

    let (_temporary, service) = service_fixture();
    let future = service.create_future(admitted_program_call()).unwrap();
    service.driver.fail_next_checkpoint_for_test();
    assert!(matches!(
        service.spawn(&future),
        Err(ExecutionServiceError::Failed(message))
            if message.contains("injected post-accept checkpoint failure")
    ));
    let snapshot = service.driver.scheduler_snapshot_for_test();
    assert_eq!(snapshot.tasks.len(), 1, "driver accepted exactly one task");
    let (task_id, task) = snapshot.tasks.iter().next().unwrap();
    assert_eq!(task.state, TaskState::Cancelled);
    assert_eq!(
        snapshot
            .events
            .iter()
            .filter(|event| matches!(&event.kind, DriverEventKind::TaskSubmitted { task_id: submitted, .. } if submitted == task_id))
            .count(),
        1,
    );
    assert_eq!(
        service.inspect_future(&future).unwrap().state,
        ExecutionHandleState::Failed,
    );
    assert!(matches!(
        service.spawn(&future),
        Err(ExecutionServiceError::Failed(_))
    ));
    assert_eq!(service.driver.scheduler_snapshot_for_test().tasks.len(), 1);
}
