mod common;
use runmat_execution::task::RetryPolicy;
use runmat_execution::{CancellationReason, ExecutionScopeId, TaskId};
use runmat_execution_runner::{
    port::BackendReport, AttemptReport, DriverCommand, DriverSnapshot,
    TaskCancellationCause as Cause,
};

fn cold(driver: &runmat_execution_runner::Driver) -> DriverSnapshot {
    let original = driver.snapshot();
    original.validate().unwrap();
    let decoded: DriverSnapshot =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, original);
    decoded
}

#[test]
fn wall_limit_records_original_attempt_and_never_relabels_it_from_a_later_report() {
    let mut f = common::fixture(1, 1);
    let mut task = common::task("wall-cause", f.scope, f.pool, RetryPolicy::Never);
    task.request.resources.max_wall_millis = 5;
    let request = common::submit(&mut f.driver, task);
    f.driver
        .handle(DriverCommand::Tick { now_millis: 4 })
        .unwrap();
    assert!(f
        .driver
        .task_execution_evidence(request.task_id)
        .unwrap()
        .task
        .cancellation
        .is_none());
    f.driver
        .handle(DriverCommand::Tick { now_millis: 5 })
        .unwrap();
    let before = cold(&f.driver).tasks[&request.task_id]
        .cancellation
        .unwrap();
    assert_eq!(
        before.cause,
        Cause::WallTime {
            attempt_id: request.id
        }
    );
    assert_eq!(before.requested_at_driver_millis, 5);
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    assert_eq!(
        cold(&f.driver).tasks[&request.task_id].cancellation,
        Some(before)
    );
    let original = f.driver.task_execution_evidence(request.task_id).unwrap();
    for forgery in 0..4 {
        let mut changed = original.clone();
        let cancellation = changed.task.cancellation.as_mut().unwrap();
        match forgery {
            0 => cancellation.requested_at_driver_millis = 4,
            1 => cancellation.requested_at_driver_millis = 6,
            2 => {
                cancellation.cause = Cause::WallTime {
                    attempt_id: runmat_execution::identity::AttemptId::derive(&[b"foreign"]),
                }
            }
            _ => {
                cancellation.cause = Cause::Provider {
                    attempt_id: request.id,
                }
            }
        }
        assert!(changed.validate().is_err());
    }
}

#[test]
fn absolute_deadline_cancels_live_and_late_dependents_with_original_dependency_identity() {
    let mut f = common::fixture(1, 1);
    let mut parent = common::task("deadline-cause", f.scope, f.pool, RetryPolicy::Never);
    parent.request.deadline_unix_millis = Some(5);
    let request = common::submit(&mut f.driver, parent);
    let mut child = common::task("dependent-cause", f.scope, f.pool, RetryPolicy::Never);
    child.dependencies.insert(request.task_id);
    let child_id = child.request.id;
    assert!(f
        .driver
        .handle(DriverCommand::Submit(Box::new(child)))
        .unwrap()
        .is_empty());
    f.driver
        .handle(DriverCommand::Tick { now_millis: 5 })
        .unwrap();
    let snapshot = cold(&f.driver);
    assert_eq!(
        snapshot.tasks[&request.task_id].cancellation.unwrap().cause,
        Cause::Deadline
    );
    assert!(snapshot.tasks[&child_id].cancellation.is_none());
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Retired,
        )))
        .unwrap();
    let snapshot = cold(&f.driver);
    assert_eq!(
        snapshot.tasks[&child_id].cancellation.unwrap().cause,
        Cause::Dependency {
            task_id: request.task_id
        }
    );
    let mut late = common::task("late-dependent-cause", f.scope, f.pool, RetryPolicy::Never);
    late.dependencies.insert(request.task_id);
    let late_id = late.request.id;
    f.driver
        .handle(DriverCommand::Submit(Box::new(late)))
        .unwrap();
    let snapshot = cold(&f.driver);
    assert_eq!(
        snapshot.tasks[&late_id].cancellation.unwrap().cause,
        Cause::Dependency {
            task_id: request.task_id
        }
    );
    let mut forged = snapshot;
    forged
        .tasks
        .get_mut(&late_id)
        .unwrap()
        .cancellation
        .as_mut()
        .unwrap()
        .cause = Cause::Dependency {
        task_id: TaskId::derive(&[b"foreign"]),
    };
    assert!(forged.validate().is_err());
}

#[test]
fn parent_scope_cancellation_retains_actual_origin_and_rejects_foreign_scope_or_reason() {
    let mut f = common::fixture(1, 1);
    let child = ExecutionScopeId::derive(&[b"typed-child"]);
    f.driver
        .handle(DriverCommand::RegisterScope {
            scope_id: child,
            parent: Some(f.scope),
        })
        .unwrap();
    let request = common::submit(
        &mut f.driver,
        common::task("scope-cause", child, f.pool, RetryPolicy::Never),
    );
    f.driver
        .handle(DriverCommand::CancelScope {
            scope_id: f.scope,
            reason: CancellationReason::User,
            now_millis: 2,
        })
        .unwrap();
    let original = cold(&f.driver);
    assert_eq!(
        original.tasks[&request.task_id].cancellation.unwrap().cause,
        Cause::Scope {
            scope_id: f.scope,
            reason: CancellationReason::User
        }
    );
    for (scope_id, reason) in [
        (
            ExecutionScopeId::derive(&[b"foreign"]),
            CancellationReason::User,
        ),
        (child, CancellationReason::User),
        (child, CancellationReason::ParentScope),
    ] {
        let mut changed = original.clone();
        changed
            .tasks
            .get_mut(&request.task_id)
            .unwrap()
            .cancellation
            .as_mut()
            .unwrap()
            .cause = Cause::Scope { scope_id, reason };
        assert!(changed.validate().is_err());
    }
    f.driver
        .handle(DriverCommand::CancelScope {
            scope_id: f.scope,
            reason: CancellationReason::ParentScope,
            now_millis: 3,
        })
        .unwrap();
    assert_eq!(
        cold(&f.driver).tasks[&request.task_id].cancellation,
        original.tasks[&request.task_id].cancellation
    );
}

#[test]
fn unsolicited_provider_cancellation_retains_original_terminal_report_identity() {
    let mut f = common::fixture(1, 1);
    let request = common::submit(
        &mut f.driver,
        common::task("provider-cause", f.scope, f.pool, RetryPolicy::Never),
    );
    f.driver
        .handle(DriverCommand::Tick { now_millis: 2 })
        .unwrap();
    f.driver
        .handle(DriverCommand::BackendReport(BackendReport::for_request(
            &request,
            AttemptReport::Cancelled,
        )))
        .unwrap();
    let original = cold(&f.driver);
    assert_eq!(
        original.tasks[&request.task_id].cancellation.unwrap().cause,
        Cause::Provider {
            attempt_id: request.id
        }
    );
    let mut changed = original;
    changed
        .tasks
        .get_mut(&request.task_id)
        .unwrap()
        .cancellation
        .as_mut()
        .unwrap()
        .cause = Cause::Deadline;
    assert!(changed.validate().is_err());
}

#[test]
fn cancellation_metadata_and_snapshot_schema_cannot_be_silently_restored_from_old_formats() {
    let f = common::fixture(1, 1);
    let mut old = f.driver.snapshot();
    old.schema_version = 2;
    assert!(old.validate().is_err());
    let mut f = common::fixture(1, 1);
    let request = common::submit(
        &mut f.driver,
        common::task("required-cause", f.scope, f.pool, RetryPolicy::Never),
    );
    let mut task = serde_json::to_value(
        f.driver
            .task_execution_evidence(request.task_id)
            .unwrap()
            .task,
    )
    .unwrap();
    task.as_object_mut().unwrap().remove("cancellation");
    assert!(serde_json::from_value::<runmat_execution_runner::task::TaskRecord>(task).is_err());
}
