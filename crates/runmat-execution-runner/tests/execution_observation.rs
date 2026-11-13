mod common;
use runmat_execution::task::RetryPolicy;
use runmat_execution_runner::port::BackendReport;
use runmat_execution_runner::{
    AttemptExecutionObservation, AttemptExecutionTermination, AttemptFailureKind, AttemptReport,
    DriverCommand,
};

#[test]
fn failed_execution_keeps_original_interval_and_transport_without_inventing_artifacts() {
    let mut fixture = common::fixture(1, 1);
    let task = common::task(
        "observed-failure",
        fixture.scope,
        fixture.pool,
        RetryPolicy::Never,
    );
    let request = common::submit(&mut fixture.driver, task);
    let observation = AttemptExecutionObservation {
        attempt_id: request.id,
        worker_id: request.worker_id,
        wall_millis: 7,
        termination: AttemptExecutionTermination::ProcessExited {
            code: Some(17),
            success: false,
        },
        local_ipc_sent_bytes: Some(123),
        local_ipc_received_bytes: Some(19),
    };
    let mut report = BackendReport::for_request(
        &request,
        AttemptReport::Failed {
            kind: AttemptFailureKind::Execution,
            message: "failed worker".into(),
        },
    );
    report.execution = Some(Box::new(observation.clone()));
    fixture
        .driver
        .handle(DriverCommand::BackendReport(report))
        .unwrap();
    let snapshot = fixture.driver.snapshot();
    let cold: runmat_execution_runner::DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&snapshot).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(
        cold.attempts[&request.id]
            .terminal
            .as_ref()
            .unwrap()
            .diagnostic
            .as_deref(),
        Some("failed worker")
    );
    assert_eq!(
        cold.attempts[&request.id]
            .terminal
            .as_ref()
            .unwrap()
            .execution
            .as_deref(),
        Some(&observation)
    );
    assert!(cold.task_resource_aggregate(request.task_id, 7).is_err());
    let mut foreign = cold;
    foreign
        .attempts
        .get_mut(&request.id)
        .unwrap()
        .terminal
        .as_mut()
        .unwrap()
        .execution
        .as_mut()
        .unwrap()
        .wall_millis = 0;
    assert!(foreign.validate().is_err());
}

#[test]
fn success_cannot_hide_nonzero_exit_or_a_different_observed_interval() {
    for corruption in 0..3 {
        let mut fixture = common::fixture(1, 1);
        let task = common::task(
            "observed-success",
            fixture.scope,
            fixture.pool,
            RetryPolicy::Never,
        );
        let request = common::submit(&mut fixture.driver, task);
        let mut observation = AttemptExecutionObservation {
            attempt_id: request.id,
            worker_id: request.worker_id,
            wall_millis: 1,
            termination: AttemptExecutionTermination::ProcessExited {
                code: Some(0),
                success: true,
            },
            local_ipc_sent_bytes: Some(123),
            local_ipc_received_bytes: Some(19),
        };
        match corruption {
            0 => {
                observation.termination = AttemptExecutionTermination::ProcessExited {
                    code: Some(17),
                    success: false,
                }
            }
            1 => observation.wall_millis = 2,
            _ => {}
        }
        let mut report = BackendReport::for_request(&request, common::success(&request));
        report.execution = Some(Box::new(observation));
        fixture
            .driver
            .handle(DriverCommand::BackendReport(report))
            .unwrap();
        let snapshot = fixture.driver.snapshot();
        assert_eq!(
            snapshot.tasks[&request.task_id].committed.is_some(),
            corruption == 2
        );
        snapshot.validate().unwrap();
        if corruption == 1 {
            // A contradictory report cannot retire the original attempt or
            // free its reservation; its backend still owes exact evidence.
            assert_eq!(
                snapshot.tasks[&request.task_id].active_attempt,
                Some(request.id)
            );
            assert!(snapshot.attempts[&request.id].terminal.is_none());
        }
        if corruption == 2 {
            let original = snapshot.task_execution_evidence(request.task_id).unwrap();
            let cold: runmat_execution_runner::TaskExecutionEvidence =
                serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
            cold.validate().unwrap();
            assert_eq!(cold, original);
            assert_eq!(
                cold.resource_aggregate(1).unwrap(),
                original.resource_aggregate(1).unwrap()
            );
            let mut invalid = snapshot;
            invalid
                .attempts
                .get_mut(&request.id)
                .unwrap()
                .terminal
                .as_mut()
                .unwrap()
                .execution
                .as_mut()
                .unwrap()
                .termination = AttemptExecutionTermination::ProcessExited {
                code: Some(17),
                success: false,
            };
            assert!(invalid.validate().is_err());
        }
    }
}

#[test]
fn cumulative_retry_reservation_rounds_each_attempt_and_rejects_unrepresentable_admission() {
    let mut fixture = common::fixture(1, 1);
    let mut task = common::task(
        "retry-envelope",
        fixture.scope,
        fixture.pool,
        RetryPolicy::TestPolicy { max_attempts: 3 },
    );
    task.request.resources.cpu_millicores = 1001;
    task.request.resources.max_wall_millis = 1;
    task.request.resources.max_artifact_bytes = 5;
    let envelope = task.retry_resource_envelope().unwrap();
    assert_eq!(envelope.maximum_attempts, 3);
    assert_eq!(envelope.maximum_reserved_usage.cpu_core_millis, 6);
    assert_eq!(envelope.maximum_reserved_usage.wall_millis, 3);
    assert_eq!(envelope.maximum_reserved_usage.memory_byte_millis, 3072);
    assert_eq!(envelope.maximum_artifact_bytes, 15);
    let accelerator = runmat_execution::resource::AcceleratorRequest::generic_compute(2).unwrap();
    task.request.resources.accelerators = vec![accelerator.clone()];
    let envelope = task.retry_resource_envelope().unwrap();
    let projected = &envelope.maximum_reserved_usage.accelerator_millis;
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].requirement, accelerator);
    assert_eq!(projected[0].device_millis, 6);
    assert_eq!(
        serde_json::from_value::<runmat_execution_runner::task::RetryResourceEnvelope>(
            serde_json::to_value(&envelope).unwrap()
        )
        .unwrap(),
        envelope
    );
    // A reservation projection grants no assigned device or measured usage.
    assert!(
        runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            task.request.resources.clone(),
            1,
            0,
            0,
            0,
        )
        .is_err()
    );
    for invalid in 0..3 {
        let mut task = task.clone();
        if invalid == 0 {
            task.request.retry = RetryPolicy::TestPolicy { max_attempts: 0 };
        } else if invalid == 1 {
            task.request.resources.max_wall_millis = u64::MAX;
        } else {
            task.request.resources.cpu_millicores = 1;
            task.request.resources.max_wall_millis = u64::MAX / 2;
        }
        let before = fixture.driver.snapshot();
        assert!(fixture
            .driver
            .handle(DriverCommand::Submit(Box::new(task)))
            .is_err());
        assert_eq!(fixture.driver.snapshot(), before);
    }
}
