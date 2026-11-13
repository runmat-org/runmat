use super::{
    request, run_process, shell_driver, TaskCompletion, TransferFailure, WorkerLaunchRequest,
};
use crate::driver::{LocalDriver, LocalProgramSubmission};
use runmat_execution::state::TaskState;
use runmat_execution::{OutputContract, RetryPolicy, TaskId};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn submit_local(driver: &Arc<LocalDriver>, label: &[u8]) -> (TaskId, Arc<TaskCompletion>) {
    let crate::protocol::WorkerRequest::Direct(request) = request() else {
        panic!("local selection fixture must be a direct request");
    };
    let task_id = TaskId::derive(&[driver.scope_id().bytes(), label]);
    let completion = driver
        .submit_with_control(
            LocalProgramSubmission {
                console: None,
                environment: runmat_execution::EnvironmentSnapshot::default(),
                task_id,
                callable: request.callable,
                invocation_context: request.context,
                recipe: request.recipe,
                artifact: request.artifact,
                inputs: request.arguments,
                outputs: OutputContract {
                    requested_outputs: request.requested_outputs,
                    owned_work: None,
                },
                retry: RetryPolicy::Never,
            },
            None,
        )
        .expect("real local driver accepted test submission");
    (task_id, completion)
}

#[test]
fn selected_queued_worker_cancellation_preserves_active_sibling() {
    let marker_directory = tempfile::tempdir().unwrap();
    let marker = marker_directory.path().join("first-worker-started");
    let script = format!("touch '{}'; sleep 30", marker.display());
    let (_temporary, driver) = shell_driver(&script);
    let (active_id, active) = submit_local(&driver, b"active");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !marker.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        marker.exists(),
        "the actual child must start before its sibling queues"
    );
    let (queued_id, queued) = submit_local(&driver, b"queued");
    let before = driver.scheduler_snapshot_for_test();
    assert_eq!(
        before.tasks.get(&queued_id).unwrap().state,
        TaskState::Ready
    );
    driver
        .cancel_task(queued_id, runmat_execution::CancellationReason::User)
        .unwrap();
    let after = driver.scheduler_snapshot_for_test();
    assert_eq!(
        after.tasks.get(&queued_id).unwrap().state,
        TaskState::Cancelled
    );
    assert_ne!(
        after.tasks.get(&active_id).unwrap().state,
        TaskState::Cancelled
    );
    assert!(queued.is_cancelled());
    assert!(!active.is_cancelled());

    driver
        .cancel_task(active_id, runmat_execution::CancellationReason::User)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while active.try_value().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(matches!(
        active.try_value(),
        Some(Err(TransferFailure::Cancelled))
    ));
}

#[tokio::test]
async fn prelaunch_cancellation_never_creates_a_child() {
    let marker_directory = tempfile::tempdir().unwrap();
    let marker = marker_directory.path().join("unexpected-child");
    let script = format!("touch '{}'", marker.display());
    let (_temporary, driver) = shell_driver(&script);
    let completion = TaskCompletion::new();
    completion.cancel();
    assert!(matches!(
        run_process(
            &driver,
            WorkerLaunchRequest::Direct(Box::new(request())),
            &completion
        )
        .await,
        Err(TransferFailure::Cancelled)
    ));
    assert!(!marker.exists());
}
