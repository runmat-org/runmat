use super::*;
use runmat_execution::identity::WorkerId;
use runmat_execution::resource::{ResourceInventory, ResourceRequest};
use runmat_execution::state::PoolState;
use runmat_execution::task::{Callable, RetryPolicy, TaskRequest};
use runmat_execution::{CancellationReason, ExecutionScopeId, OutputContract, PoolId, TaskId};
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptReport, AttemptSuccess, Driver, DriverAction, DriverCommand, DriverConfig, PoolSpec,
    TaskSubmission, WorkerSpec,
};
use std::collections::BTreeSet;

fn assigned() -> (Driver, AttemptRequest) {
    let scope = ExecutionScopeId::derive(&[b"completion"]);
    let pool = PoolId::derive(&[b"completion"]);
    let worker = WorkerId::derive(&[b"completion"]);
    let inventory = ResourceInventory {
        cpu_millicores: 1000,
        memory_bytes: 1024,
        scratch_bytes: 1024,
        accelerators: Vec::new(),
        capabilities: BTreeSet::new(),
    };
    let mut driver = Driver::new(DriverConfig::default(), 1).unwrap();
    driver
        .handle(DriverCommand::RegisterScope {
            scope_id: scope,
            parent: None,
        })
        .unwrap();
    driver
        .handle(DriverCommand::CreatePool(PoolSpec {
            id: pool,
            min_workers: 1,
            max_workers: 1,
            max_in_flight: 1,
            resource_limit: inventory.clone(),
        }))
        .unwrap();
    driver
        .handle(DriverCommand::SetPoolState {
            pool_id: pool,
            state: PoolState::Ready,
        })
        .unwrap();
    driver
        .handle(DriverCommand::RegisterWorker(WorkerSpec {
            host: crate::test_support::host_inventory(),
            id: worker,
            pool_id: pool,
            resources: inventory,
        }))
        .unwrap();
    let request = TaskRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        id: TaskId::derive(&[b"completion"]),
        scope_id: scope,
        pool_id: pool,
        schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
        executable: runmat_execution::TaskExecutableIdentity::ProgramArtifact(ArtifactId::derive(
            &[b"completion"],
        )),
        host: crate::test_support::host_requirement(),
        invocation_context: runmat_execution::ProgramInvocationContext::Direct,
        callable: Callable::for_program(
            "test",
            &runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(0),
                Some("completion".into()),
            ),
        ),
        inputs: Vec::new(),
        outputs: OutputContract {
            owned_work: None,
            requested_outputs: 0,
        },
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 1024,
            scratch_bytes: 1024,
            max_wall_millis: 1000,
            max_artifact_bytes: 1024,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: BTreeSet::new(),
        },
        retry: RetryPolicy::Never,
        deadline_unix_millis: None,
    };
    let attempt = driver
        .handle(DriverCommand::Submit(Box::new(TaskSubmission {
            request,
            dependencies: BTreeSet::new(),
            priority: 0,
        })))
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    (driver, attempt)
}

#[test]
fn completion_publishes_result_and_history_once() {
    use std::future::Future;
    use std::task::{Context, Poll};
    struct Wake(std::sync::atomic::AtomicUsize);
    impl futures::task::ArcWake for Wake {
        fn wake_by_ref(this: &Arc<Self>) {
            this.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let completion = Arc::new(TaskCompletion::new());
    let wake = Arc::new(Wake(std::sync::atomic::AtomicUsize::new(0)));
    let waker = futures::task::waker(wake.clone());
    let mut waiting = Box::pin(completion.clone().wait_terminal());
    assert!(waiting
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    assert!(completion.try_value().is_none());
    assert!(completion.try_attempt_history().is_none());
    completion.complete(
        CompletedTask {
            result: Err(TransferFailure::Execution("original".into())),
            evidence: None,
        },
        NativeCheckpointEvidence::Durable,
    );
    completion.complete(
        CompletedTask {
            result: Err(TransferFailure::Execution("late".into())),
            evidence: None,
        },
        NativeCheckpointEvidence::Durable,
    );
    assert_eq!(
        completion.try_value(),
        Some(Err(TransferFailure::Execution("original".into())))
    );
    assert_eq!(completion.try_attempt_history(), Some(Vec::new()));
    assert!(
        wake.0.load(Ordering::SeqCst) > 0,
        "terminal publication must wake, not require a polling timer"
    );
    assert!(matches!(
        waiting.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Ready(())
    ));
    futures::executor::block_on(completion.clone().wait_terminal());
}

#[test]
fn late_worker_success_cannot_override_scheduler_cancellation() {
    let (mut driver, request) = assigned();
    let success = AttemptSuccess::new(&request, Vec::new(), Vec::new(), 1, 0, 0).unwrap();
    driver
        .handle(DriverCommand::CancelScope {
            scope_id: request.scope_id,
            reason: CancellationReason::User,
            now_millis: 1,
        })
        .unwrap();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded {
                result: success.clone(),
            },
        )))
        .unwrap();
    let completed = from_snapshot(&driver.snapshot(), &request, Ok(success.clone())).unwrap();
    assert!(matches!(
        completed.result,
        Err(TransferFailure::RetirementUnverified(_))
    ));
    let evidence = completed.evidence.as_ref().unwrap();
    assert_eq!(evidence.attempts.len(), 1);
    assert!(evidence.attempts[0].terminal.is_none());
    let unresolved = evidence.attempts[0].unverified_retirement.as_ref().unwrap();
    assert_eq!(unresolved.resources.as_deref(), success.resources());
    assert!(unresolved.retired_at_driver_millis.is_none());
    assert!(evidence.resource_aggregate(1).is_err());
    assert!(driver.snapshot().tasks[&request.task_id]
        .committed
        .is_none());
    let original_cancellation = evidence.task.cancellation.unwrap();
    let completion = TaskCompletion::new();
    completion.complete(completed, NativeCheckpointEvidence::Durable);
    let completion = completion.try_native_completion().unwrap();
    assert!(
        matches!(completion.delivery_result(), Err(crate::NativeExecutionError::TaskCancelled {
        task_id, cancellation, quiescence: crate::NativeTaskQuiescence::Unconfirmed,
    }) if task_id == request.task_id && cancellation == original_cancellation)
    );
}

#[test]
fn committed_original_result_wins_over_a_different_raw_response() {
    let (mut driver, request) = assigned();
    let success = AttemptSuccess::new(&request, Vec::new(), Vec::new(), 1, 0, 0).unwrap();
    driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Succeeded {
                result: success.clone(),
            },
        )))
        .unwrap();
    let completed = from_snapshot(
        &driver.snapshot(),
        &request,
        Err(TransferFailure::Execution("stale raw error".into())),
    )
    .unwrap();
    assert_eq!(completed.result, Ok(success));
    assert_eq!(completed.evidence.as_ref().unwrap().attempts.len(), 1);
    let mut invalid = driver.snapshot();
    invalid.attempts.get_mut(&request.id).unwrap().terminal = None;
    assert!(from_snapshot(
        &invalid,
        &request,
        Ok(AttemptSuccess::Values {
            outputs: Vec::new(),
            result_objects: Vec::new(),
            resources: None,
        })
    )
    .unwrap()
    .result
    .is_err());
}

#[test]
fn uncertain_checkpoint_blocks_delivery_without_rewriting_original_execution() {
    for checkpoint in [
        NativeCheckpointEvidence::NotPublished {
            diagnostic: "injected write failure".into(),
        },
        NativeCheckpointEvidence::DurabilityIndeterminate {
            diagnostic: "injected directory flush failure".into(),
        },
    ] {
        let (mut driver, request) = assigned();
        let success = AttemptSuccess::new(&request, Vec::new(), Vec::new(), 1, 0, 0).unwrap();
        driver
            .handle(DriverCommand::BackendReport(BackendReport::for_request(
                &request,
                AttemptReport::Succeeded {
                    result: success.clone(),
                },
            )))
            .unwrap();
        let snapshot = driver.snapshot();
        let original = from_snapshot(&snapshot, &request, Ok(success.clone())).unwrap();
        let evidence = original.evidence.clone().unwrap();
        let history = evidence.attempts.clone();
        let completion = TaskCompletion::new();
        completion.complete(original, checkpoint.clone());
        assert_eq!(
            completion.try_value(),
            Some(Err(TransferFailure::Infrastructure(
                checkpoint.delivery_error().unwrap()
            )))
        );
        assert_eq!(completion.try_scheduler_result(), Some(Ok(success)));
        assert_eq!(completion.try_checkpoint_evidence(), Some(checkpoint));
        assert_eq!(completion.try_attempt_history(), Some(history));
        assert_eq!(completion.try_execution_evidence(), Some(evidence));
        assert_eq!(snapshot.tasks[&request.task_id].state, TaskState::Succeeded);
        snapshot.validate().unwrap();
    }
}

mod analysis;
mod cancellation;
