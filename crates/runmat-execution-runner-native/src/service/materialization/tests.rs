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
        local_fea_organization: Default::default(),
        executable: "/bin/sh".into(),
        worker_arguments: vec!["-c".into(), "exit 0".into()],
        max_workers: 1,
        local_cpu_capacity_millicores: None,
        local_memory_capacity_bytes: None,
        authored_worker_memory_bytes: None,
        worker_memory_bytes: 1024 * 1024 * 1024,
        worker_scratch_bytes: 1024 * 1024 * 1024,
        worker_cpu_millicores: 1_000,
        meshing_subtask_cache_root: None,
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
        spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
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
        spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
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
    let outputs = OutputContract {
        requested_outputs: 1,
        owned_work: None,
    };
    let mut futures = crate::service::FutureTable::default();
    futures.issue(id, outputs.clone(), FutureState::Deferred(Rc::new(call)));
    let state = State {
        next_future: 1,
        next_task: 0,
        next_materialization: 0,
        futures,
        tasks: HashMap::new(),
        durable_jobs: HashMap::new(),
        cancelling_graphs: Vec::new(),
        pool_generation: 1,
        pool_open: true,
    };
    (
        Mutex::new(state),
        FutureHandle {
            id,
            scope_id,
            outputs,
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

#[cfg(unix)]
#[test]
fn durable_replay_retains_one_cache_and_exposes_its_values_to_alias_inspection() {
    use crate::durable::SubmittedJob;
    use crate::service::job_receipt::JobResultAuthority;
    use crate::supervisor::JobSubmissionDisposition;
    use runmat_execution::{JobHandle, JobId, RunId};
    use runmat_runtime::execution::{RootedValueSequence, RuntimeExecutionServices};

    let (_temporary, service) = service_fixture();
    let handle = JobHandle {
        id: JobId::derive(&[b"durable-cache-owner"]),
        run_id: RunId::derive(&[b"durable-cache-owner"]),
        generation: 1,
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
    };
    let submitted = |disposition| SubmittedJob {
        handle: handle.clone(),
        retain_until_unix_millis: u64::MAX,
        disposition,
    };
    let first = service
        .retain_durable_receipt(
            submitted(JobSubmissionDisposition::Created),
            JobResultAuthority::Direct,
        )
        .unwrap();
    *first.terminal.borrow_mut() = Some(RootedValueSequence::new(
        runmat_value::ValueSequence::single(Value::Num(73.0)).unwrap(),
    ));
    let replay = service
        .retain_durable_receipt(
            submitted(JobSubmissionDisposition::Duplicate),
            JobResultAuthority::Direct,
        )
        .unwrap();
    assert!(Rc::ptr_eq(&first, &replay));
    let mut observed = Vec::new();
    assert_eq!(
        service.visit_retained_values(&mut |value| {
            observed.push(value.clone());
            Some(())
        }),
        Some(())
    );
    assert_eq!(observed, vec![Value::Num(73.0)]);
    let active_publication = first.terminal.borrow_mut();
    assert_eq!(service.visit_retained_values(&mut |_| Some(())), None);
    drop(active_publication);

    let mut changed = submitted(JobSubmissionDisposition::Duplicate);
    changed.handle.generation += 1;
    assert!(matches!(
        service.retain_durable_receipt(changed, JobResultAuthority::Direct),
        Err(ExecutionServiceError::ForeignScope)
    ));
    let mut changed = submitted(JobSubmissionDisposition::Duplicate);
    changed.retain_until_unix_millis -= 1;
    assert!(matches!(
        service.retain_durable_receipt(changed, JobResultAuthority::Direct),
        Err(ExecutionServiceError::ForeignScope)
    ));
    assert_eq!(service.state.lock().unwrap().durable_jobs.len(), 1);
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
    use runmat_execution::{CancellationReason, ExecutionHandleState, TaskHandle, TaskId};
    use runmat_runtime::execution::{
        AwaitAction, RootedValueSequence, RuntimeExecutionServices, TaskRetention,
    };

    let (_temporary, service) = service_fixture();
    let result =
        RootedValueSequence::new(runmat_value::ValueSequence::single(Value::Num(7.0)).unwrap())
            .unwrap();
    // A caller-executed future has no worker submission. Bind its ordinary
    // public task alias in the same private table used by materialization.
    let alias = |future: &FutureHandle| {
        let task = TaskHandle {
            id: TaskId::derive(&[future.id.bytes(), b"terminal-alias"]),
            scope_id: future.scope_id,
            generation: 1,
            outputs: future.outputs.clone(),
        };
        service.state.lock().unwrap().tasks.insert(
            task.id,
            super::TaskRecord {
                future_id: future.id,
                generation: task.generation,
                read: false,
                claimed: false,
                completion_order: None,
                retention: TaskRetention::Refetchable,
                callback_completion: None,
            },
        );
        task
    };
    for terminal in [
        Ok(result),
        Err(ExecutionServiceError::Failed(
            "original caller failure".into(),
        )),
    ] {
        let future = service.create_future(admitted_program_call()).unwrap();
        let task = alias(&future);
        assert!(service
            .complete_future((&future).into(), terminal.clone())
            .is_err());
        assert!(matches!(
            service
                .begin_await(Value::Future(future.clone().into()))
                .unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        service
            .complete_future((&future).into(), terminal.clone())
            .unwrap();
        assert!(service
            .complete_future((&future).into(), Err(ExecutionServiceError::Cancelled))
            .is_err());
        for handle in [
            Value::Future(future.clone().into()),
            Value::Task(task.clone().into()),
        ] {
            service.cancel(&handle, CancellationReason::User).unwrap();
            service.drain_scope(CancellationReason::Shutdown);
            assert_eq!(
                service.begin_await(handle),
                terminal.clone().map(AwaitAction::Completed)
            );
        }
        let expected = if terminal.is_ok() {
            ExecutionHandleState::Finished
        } else {
            ExecutionHandleState::Failed
        };
        assert_eq!(service.inspect_future(&future).unwrap().state, expected);
        assert_eq!(service.inspect_task(&task).unwrap().state, expected);
    }

    let active = service.create_future(admitted_program_call()).unwrap();
    let task = alias(&active);
    assert!(matches!(
        service
            .begin_await(Value::Task(task.clone().into()))
            .unwrap(),
        AwaitAction::ExecuteFuture { .. }
    ));
    service
        .cancel(&Value::Task(task.clone().into()), CancellationReason::User)
        .unwrap();
    assert_eq!(
        service.complete_future(
            (&active).into(),
            Err(ExecutionServiceError::Failed("late caller".into()))
        ),
        Err(ExecutionServiceError::Cancelled)
    );
    for handle in [
        Value::Future(active.into()),
        Value::Task(task.clone().into()),
    ] {
        service
            .cancel(&handle, CancellationReason::Shutdown)
            .unwrap();
        assert_eq!(
            service.begin_await(handle),
            Err(ExecutionServiceError::Cancelled)
        );
    }
    assert_eq!(
        service.inspect_task(&task).unwrap().state,
        ExecutionHandleState::Cancelled
    );
}

#[cfg(unix)]
#[test]
fn failed_checkpoint_publishes_no_task_and_cannot_submit_twice() {
    use runmat_execution::ExecutionHandleState;
    use runmat_execution_runner::driver::DriverEventKind;
    use runmat_runtime::execution::RuntimeExecutionServices;

    let (_temporary, service) = service_fixture();
    let future = service.create_future(admitted_program_call()).unwrap();
    service.driver.fail_next_checkpoint_for_test();
    assert!(matches!(
        service.spawn(&future),
        Err(ExecutionServiceError::Failed(message))
            if message.contains("injected checkpoint publication failure")
    ));
    let snapshot = service.driver.scheduler_snapshot_for_test();
    assert!(
        snapshot.tasks.is_empty(),
        "unpublished admission is rolled back"
    );
    assert!(
        snapshot.attempts.is_empty(),
        "no attempt may launch before publication"
    );
    assert!(!snapshot
        .events
        .iter()
        .any(|event| matches!(&event.kind, DriverEventKind::TaskSubmitted { .. })));
    assert_eq!(
        service.inspect_future(&future).unwrap().state,
        ExecutionHandleState::Failed,
    );
    assert!(matches!(
        service.spawn(&future),
        Err(ExecutionServiceError::Failed(_))
    ));
    assert!(service
        .driver
        .scheduler_snapshot_for_test()
        .tasks
        .is_empty());
}

#[cfg(unix)]
#[test]
fn callback_waves_release_only_internal_records_after_rooted_result_handoff() {
    use runmat_execution::{TaskHandle, TaskId};
    use runmat_runtime::execution::{RootedValueSequence, RuntimeExecutionServices, TaskRetention};
    runmat_gc::gc_test_context(|| {
        let (_temporary, service) = service_fixture();
        let public_future = service.create_future(admitted_program_call()).unwrap();
        let public_task = TaskHandle {
            id: TaskId::derive(&[b"refetchable"]),
            scope_id: service.scope_id,
            generation: 1,
            outputs: public_future.outputs.clone(),
        };
        service.state.lock().unwrap().tasks.insert(
            public_task.id,
            super::TaskRecord {
                future_id: public_future.id,
                generation: 1,
                read: false,
                claimed: false,
                completion_order: None,
                retention: TaskRetention::Refetchable,
                callback_completion: None,
            },
        );
        assert!(service
            .release_callback_batch(std::slice::from_ref(&public_task))
            .is_err());
        for wave in 0..3u64 {
            let allocation = runmat_gc::gc_allocate_rooted(Value::Num(wave as f64)).unwrap();
            let handle = allocation.handle();
            let value = Value::HandleObject(runmat_value::HandleRef {
                class_name: "RetainedCallbackValue".into(),
                target: handle,
                valid: true,
            });
            let result =
                RootedValueSequence::new(runmat_value::ValueSequence::single(value).unwrap())
                    .unwrap();
            drop(allocation);
            let future_id = FutureId::derive(&[b"callback", &wave.to_be_bytes()]);
            let task = TaskHandle {
                id: TaskId::derive(&[b"callback", &wave.to_be_bytes()]),
                scope_id: service.scope_id,
                generation: 1,
                outputs: public_future.outputs.clone(),
            };
            {
                let mut state = service.state.lock().unwrap();
                state.futures.issue(
                    future_id,
                    public_future.outputs.clone(),
                    FutureState::Completed(Ok(result.clone())),
                );
                state.tasks.insert(
                    task.id,
                    super::TaskRecord {
                        future_id,
                        generation: 1,
                        read: false,
                        claimed: false,
                        completion_order: Some(wave),
                        retention: TaskRetention::CallbackBatch,
                        callback_completion: Some(
                            crate::driver::LocalDriver::settled_callback_completion_for_test(),
                        ),
                    },
                );
            }
            service
                .release_callback_batch(std::slice::from_ref(&task))
                .unwrap();
            {
                let state = service.state.lock().unwrap();
                assert_eq!(state.tasks.len(), 1);
                assert_eq!(state.futures.len(), 1);
            }
            runmat_gc::gc_collect_major().unwrap();
            assert!(
                runmat_gc::gc_with_value(&handle, |_| ()).is_ok(),
                "the returned result owns roots after service retirement"
            );
            let rooted = result.try_into_single().unwrap();
            assert!(matches!(rooted.value(), Value::HandleObject(_)));
            drop(rooted);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
        }
        // A private Future that fails before atomic publication has no task
        // record from which a later release could discover it.
        let pool = service
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .unwrap();
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        // Opaque program bytes are validated by their execution owner, not
        // by this transport boundary. Fail the actual publication boundary
        // to exercise retirement before a task record becomes observable.
        service.driver.fail_next_checkpoint_for_test();
        let failure = service
            .submit_callback_batch(
                vec![admitted_program_call()],
                &pool.handle,
                cancellation.token(),
            )
            .unwrap_err();
        assert!(
            matches!(failure, ExecutionServiceError::Infrastructure(message)
            if message.contains("injected checkpoint publication failure"))
        );
        {
            let state = service.state.lock().unwrap();
            assert_eq!(state.futures.len(), 1, "unpublished private Future retired");
            assert_eq!(state.tasks.len(), 1);
        }
        let snapshot = service.driver.scheduler_snapshot_for_test();
        assert!(snapshot.tasks.is_empty());
        assert!(snapshot.attempts.is_empty());
        assert!(!snapshot.events.iter().any(|event| matches!(
            event.kind,
            runmat_execution_runner::driver::DriverEventKind::TaskSubmitted { .. }
        )));
        assert!(
            service.inspect_future(&public_future).is_ok(),
            "ordinary refetchable Future is untouched"
        );
    });
}

#[test]
fn reservation_rejects_a_handle_whose_output_contract_differs_from_the_issued_one() {
    let (state, future) = fixture();
    let mut altered = future.clone();
    altered.outputs.requested_outputs += 1;
    assert!(matches!(
        reserve(&state, &altered),
        Err(runmat_runtime::execution::ExecutionServiceError::InvalidOutputContract)
    ));
    assert!(matches!(
        state.lock().unwrap().futures.get(&future.id),
        Some(FutureState::Deferred(_))
    ));
    reserve(&state, &future).unwrap();
}
