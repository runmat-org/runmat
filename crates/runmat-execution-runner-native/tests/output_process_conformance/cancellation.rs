//! Direct native task cancellation must retain the original cause and observed exit.
use super::*;
use runmat_execution::CancellationReason;
use runmat_execution_runner::{AttemptExecutionTermination, DriverSnapshot, TaskCancellationCause};
use runmat_runtime::execution::ExecutionServiceError;
use std::path::Path;

pub(super) const WORKER_MARKER: &str = "--original-output-blocked-worker";

pub(super) async fn blocked_worker() {
    let bytes = runmat_process_host::ipc::read_payload(
        &mut tokio::io::stdin(),
        runmat_process_host::ipc::FrameLimits {
            max_message_bytes: 64 * 1024 * 1024,
        },
    )
    .await
    .unwrap();
    let request: runmat_execution_artifact::ProgramExecutionRequest =
        serde_json::from_slice(&bytes).unwrap();
    request.validate_for_portable_host().unwrap();
    let objects =
        std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV).unwrap();
    let root = Path::new(&objects).parent().unwrap();
    runmat_filesystem::NativeFsProvider
        .atomic_replace_sync(
            &root.join("original-worker-ready"),
            request.artifact.id.0.bytes(),
        )
        .unwrap();
    // No cooperative response: the actual provider must terminate and reap the process.
    std::thread::sleep(Duration::from_secs(60));
}

pub(super) async fn verify(original: &NativeExecutionConfig) {
    let mut config = original.clone();
    config.worker_arguments = vec![WORKER_MARKER.into()];
    let service = NativeExecutionService::new(config.clone()).unwrap();
    let future = service.create_future(call(1)).unwrap();
    let task = service.spawn(&future).unwrap();
    let root = config.store_root.join(task.scope_id.to_string());
    tokio::time::timeout(Duration::from_secs(30), async {
        while !root.join("original-worker-ready").exists() {
            assert!(matches!(
                service.begin_await(Value::Task(task.clone())).unwrap(),
                AwaitAction::Pending(_)
            ));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original worker did not accept its request");
    service
        .cancel(&Value::Task(task.clone()), CancellationReason::Quota)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match service.begin_await(Value::Future(future.clone())) {
                Ok(AwaitAction::Pending(_)) => tokio::time::sleep(Duration::from_millis(10)).await,
                Err(ExecutionServiceError::Cancelled) => break,
                other => panic!("original running cancellation returned {other:?}"),
            }
        }
    })
    .await
    .expect("original cancellation did not observe worker termination");
    let bytes = std::fs::read(root.join("checkpoints/driver.json")).unwrap();
    let cold: DriverSnapshot = serde_json::from_slice(&bytes).unwrap();
    cold.validate().unwrap();
    let evidence = cold.task_execution_evidence(task.id).unwrap();
    assert_eq!(
        evidence.task.state,
        runmat_execution::state::TaskState::Cancelled
    );
    assert_eq!(
        evidence.task.cancellation.unwrap().cause,
        TaskCancellationCause::Task {
            task_id: task.id,
            reason: CancellationReason::Quota,
        }
    );
    assert_eq!(evidence.attempts.len(), 1);
    assert!(matches!(
        evidence.attempts[0]
            .terminal
            .as_ref()
            .unwrap()
            .execution
            .as_ref()
            .unwrap()
            .termination,
        AttemptExecutionTermination::ProcessExited { success: false, .. }
    ));
    service
        .cancel(&Value::Future(future), CancellationReason::User)
        .unwrap();
    let repeated: DriverSnapshot =
        serde_json::from_slice(&std::fs::read(root.join("checkpoints/driver.json")).unwrap())
            .unwrap();
    repeated.validate().unwrap();
    assert_eq!(
        repeated.tasks[&task.id].cancellation,
        evidence.task.cancellation
    );
    println!("original running task cancellation retained its cause, observed process exit and cold checkpoint");
}
