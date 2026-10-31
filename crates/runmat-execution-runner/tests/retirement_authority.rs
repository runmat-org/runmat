mod common;

use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::CancellationReason;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::recovery::DriverCheckpoint;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand};

#[test]
fn uncertainty_retains_capacity_until_exact_proof_without_replaying_original_task() {
    let mut f = common::fixture(2, 1);
    let request = common::submit(
        &mut f.driver,
        common::task(
            "original",
            f.scope,
            f.pool,
            RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
        ),
    );
    let original = f.driver.snapshot();
    assert!(f
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::RetirementUnverified {
                message: String::new()
            },
        )))
        .is_err());
    assert_eq!(f.driver.snapshot(), original);
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::RetirementUnverified {
                message: "original retirement failure".into(),
            },
        )))
        .unwrap();
    let saved = f.driver.snapshot();
    saved.validate().unwrap();
    assert_eq!(
        saved.tasks[&request.task_id].state,
        TaskState::Indeterminate
    );
    assert_eq!(
        saved.tasks[&request.task_id].active_attempt,
        Some(request.id)
    );
    assert!(saved.attempts[&request.id].terminal.is_none());
    let mut contradictory = BackendReport::for_request(&request, common::success(&request));
    contradictory.execution = Some(Box::new(
        runmat_execution_runner::AttemptExecutionObservation {
            attempt_id: request.id,
            worker_id: request.worker_id,
            wall_millis: 2,
            termination: runmat_execution_runner::AttemptExecutionTermination::ProcessExited {
                code: Some(0),
                success: true,
            },
            local_ipc_sent_bytes: Some(0),
            local_ipc_received_bytes: Some(0),
        },
    ));
    f.driver
        .handle(DriverCommand::BackendReport(contradictory))
        .unwrap();
    assert_eq!(f.driver.snapshot().attempts, saved.attempts);
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            common::success(&request),
        )))
        .unwrap();
    let measured = f.driver.snapshot();
    assert!(measured.tasks[&request.task_id].committed.is_none());
    assert_eq!(
        measured.attempts[&request.id]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .resources
            .as_ref()
            .unwrap()
            .resources
            .usage
            .wall_millis,
        1
    );
    let failure = runmat_execution::ProgramRuntimeFailure {
        message: "original numeric error arrived after loss".into(),
        identifier: Some("Test:Original".into()),
        span: None,
        builtin: None,
        task_id: None,
        call_frames: Vec::new(),
        call_frames_elided: 0,
        call_stack: Vec::new(),
        phase: None,
    };
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::RuntimeFailed {
                failure: failure.clone(),
            },
        )))
        .unwrap();
    let enriched = f.driver.snapshot();
    let uncertainty = enriched.attempts[&request.id]
        .unverified_retirement
        .as_ref()
        .unwrap();
    assert_eq!(uncertainty.failure.as_deref(), Some(&failure));
    assert_eq!(uncertainty.diagnostic, "original retirement failure");
    enriched.validate().unwrap();
    assert_eq!(
        saved.pools[&f.pool].allocated,
        original.pools[&f.pool].allocated
    );
    let mut replacement = saved.pools[&f.pool].workers[&request.worker_id]
        .spec
        .clone();
    replacement.id = runmat_execution::identity::WorkerId::derive(&[b"unproved replacement"]);
    assert!(f
        .driver
        .handle(DriverCommand::RegisterWorker(replacement))
        .is_err());
    let cold = serde_json::from_slice::<runmat_execution_runner::DriverSnapshot>(
        &serde_json::to_vec(&saved).unwrap(),
    )
    .unwrap();
    cold.validate().unwrap();
    let mut forged = cold.clone();
    forged.pools.get_mut(&f.pool).unwrap().allocated = Default::default();
    assert!(forged.validate().is_err());
    let queued = common::task("other", f.scope, f.pool, RetryPolicy::Never);
    let queued_id = queued.request.id;
    assert!(!f
        .driver
        .handle(DriverCommand::Submit(Box::new(queued)))
        .unwrap()
        .iter()
        .any(|action| matches!(action, DriverAction::Launch(_))));
    let mut wrong = BackendReport::for_request(&request, AttemptReport::Retired);
    wrong.driver_fence += 1;
    assert!(!f.driver.accepts_backend_report(&wrong));
    let actions = f
        .driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    assert!(actions.iter().any(
        |action| matches!(action, DriverAction::Launch(attempt) if attempt.task_id == queued_id)
    ));
    let retired = f.driver.snapshot();
    retired.validate().unwrap();
    assert_eq!(retired.tasks[&request.task_id].state, TaskState::Failed);
    assert_eq!(
        retired.attempts[&request.id]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .failure
            .as_deref(),
        Some(&failure)
    );
    assert_eq!(retired.tasks[&request.task_id].attempt_count, 1);
    assert!(retired.attempts[&request.id]
        .unverified_retirement
        .as_ref()
        .unwrap()
        .retired_at_driver_millis
        .is_some());
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    assert_eq!(f.driver.snapshot().pools, retired.pools);
    let aggregate = f
        .driver
        .task_execution_evidence(request.task_id)
        .unwrap()
        .resource_aggregate(1)
        .unwrap();
    assert_eq!(aggregate.attempts.len(), 1);
    assert_eq!(aggregate.attempts[0].attempt_id, request.id);
    assert_eq!(
        aggregate.attempts[0],
        *measured.attempts[&request.id]
            .unverified_retirement
            .as_ref()
            .unwrap()
            .resources
            .as_deref()
            .unwrap()
    );
}

#[test]
fn loss_and_replacement_fence_preserve_original_reservation_until_old_attempt_proof() {
    let mut f = common::fixture(1, 1);
    let request = common::submit(
        &mut f.driver,
        common::task(
            "lost",
            f.scope,
            f.pool,
            RetryPolicy::ExplicitlyIdempotent { max_attempts: 2 },
        ),
    );
    f.driver
        .handle(DriverCommand::DrainWorker(request.worker_id))
        .unwrap();
    let draining = DriverCheckpoint::new(1, f.driver.snapshot())
        .restore()
        .unwrap()
        .snapshot();
    assert_eq!(
        draining.pools[&f.pool].workers[&request.worker_id].lifecycle,
        runmat_execution_runner::pool::WorkerLifecycle::Lost
    );
    f.driver
        .handle(DriverCommand::WorkerLost(request.worker_id))
        .unwrap();
    let saved = f.driver.snapshot();
    let mut restored = DriverCheckpoint::new(1, saved.clone()).restore().unwrap();
    assert_eq!(restored.snapshot().pools, saved.pools);
    assert!(
        !restored.accepts_backend_report(&BackendReport::for_request(
            &request,
            common::success(&request)
        ))
    );
    restored
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    let retired = restored.snapshot();
    retired.validate().unwrap();
    assert_eq!(retired.pools[&f.pool].active_attempts, 0);
    assert_eq!(retired.pools[&f.pool].allocated, Default::default());
    assert_eq!(
        retired.tasks[&request.task_id].state,
        TaskState::Indeterminate
    );
}

#[test]
fn nearest_deadline_escalates_without_fabricating_direct_retirement() {
    let mut f = common::fixture(1, 1);
    let mut task = common::task("deadline", f.scope, f.pool, RetryPolicy::Never);
    task.request.resources.max_wall_millis = 5;
    let request = common::submit(&mut f.driver, task);
    assert_eq!(f.driver.next_timer_deadline_millis(), Some(5));
    f.driver
        .handle(DriverCommand::Tick { now_millis: 5 })
        .unwrap();
    assert_eq!(f.driver.next_timer_deadline_millis(), Some(2_005));
    f.driver
        .handle(DriverCommand::Tick { now_millis: 2_005 })
        .unwrap();
    assert_eq!(f.driver.next_timer_deadline_millis(), Some(10_005));
    f.driver
        .handle(DriverCommand::Tick { now_millis: 10_005 })
        .unwrap();
    assert_eq!(f.driver.next_timer_deadline_millis(), None);
    assert_eq!(f.driver.snapshot().pools[&f.pool].active_attempts, 1);
    let cancellation = f
        .driver
        .task_record(request.task_id)
        .unwrap()
        .cancellation
        .unwrap();
    assert_eq!(
        cancellation.cause.reason_for_attempt(&request).unwrap(),
        CancellationReason::Deadline
    );
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    assert_eq!(
        f.driver.task_record(request.task_id).unwrap().state,
        TaskState::Cancelled
    );
    f.driver.snapshot().validate().unwrap();
}
