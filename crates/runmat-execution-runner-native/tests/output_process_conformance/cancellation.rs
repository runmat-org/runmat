//! Direct native task cancellation must retain the original cause and observed exit.
use super::*;
use runmat_execution::CancellationReason;
use runmat_execution_runner::{AttemptExecutionTermination, DriverSnapshot, TaskCancellationCause};
use runmat_runtime::execution::ExecutionServiceError;
use std::path::Path;

pub(super) const WORKER_MARKER: &str = "--original-output-blocked-worker";

pub(super) async fn blocked_worker() {
    let mut control = runmat_process_host::ipc::private_control::connect_host()
        .await
        .expect("original worker private control connection");
    let bytes = runmat_process_host::ipc::read_payload(
        &mut control.reader,
        runmat_process_host::ipc::FrameLimits {
            max_message_bytes: 64 * 1024 * 1024,
        },
    )
    .await
    .unwrap();
    // This intentionally noncooperative process observes launch delivery, not
    // execution authority: only the real worker may admit a staged graph.
    let launch: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let launch = launch.as_object().expect("one staged launch envelope");
    assert_eq!(launch.len(), 1);
    let reference = launch["staged_graph_v10"]
        .as_object()
        .expect("actual parent-issued staged graph reference");
    assert_eq!(reference.len(), 10);
    for field in [
        "fea_parent_admission",
        "artifact_id",
        "stored_digest",
        "stored_length",
        "assignment",
        "context",
        "requested_outputs",
        "resources",
        "console_budget",
    ] {
        assert!(
            !reference[field].is_null(),
            "missing launch reference {field}"
        );
    }
    // A non-FEA program carries an explicit null FEA grant.
    assert!(reference["fea_grant"].is_null() && reference.contains_key("fea_grant"));
    let resources: runmat_execution::resource::ResourceRequest =
        serde_json::from_value(reference["resources"].clone()).unwrap();
    resources.validate().unwrap();
    let record = runmat_execution::console::ConsoleRecord {
        sequence: 1,
        stream: runmat_execution::console::ConsoleStream::Stdout,
        text: "original before termination".into(),
    };
    let message = runmat_execution_runner_native::WorkerProcessMessage::Console { record };
    runmat_process_host::ipc::write_payload(
        &mut control.writer,
        &serde_json::to_vec(&message).unwrap(),
        runmat_process_host::ipc::FrameLimits {
            max_message_bytes: 64 * 1024 * 1024,
        },
    )
    .await
    .unwrap();
    let objects =
        std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV).unwrap();
    let root = Path::new(&objects).parent().unwrap();
    runmat_filesystem::NativeFsProvider
        .atomic_replace_sync(&root.join("original-worker-ready"), b"launch-delivered")
        .unwrap();
    // No cooperative response: the actual provider must terminate and reap the process.
    std::thread::sleep(Duration::from_secs(60));
}

pub(super) async fn verify(original: &NativeExecutionConfig) {
    let mut config = original.clone();
    config.worker_arguments = vec![WORKER_MARKER.into()];
    let service = NativeExecutionService::new(config.clone()).unwrap();
    let original = call(1).await;
    let future = service.create_future(original.call.clone()).unwrap();
    let task = service.spawn(&future).unwrap();
    let root = config.store_root.join(task.scope_id.to_string());
    tokio::time::timeout(Duration::from_secs(30), async {
        while !root.join("original-worker-ready").exists() {
            assert!(matches!(
                service
                    .begin_await_in_context(Value::Task(task.clone().into()), &original.runtime)
                    .unwrap(),
                AwaitAction::Pending(_)
            ));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original worker did not accept its request");
    service
        .cancel(&Value::Task(task.clone().into()), CancellationReason::Quota)
        .unwrap();
    assert!(matches!(
        await_in_context(
            &service,
            Value::Future(future.clone().into()),
            &original.runtime,
        )
        .await,
        Err(ExecutionServiceError::Cancelled)
    ));
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
    let prefix = evidence.attempts[0]
        .terminal
        .as_ref()
        .unwrap()
        .console
        .as_ref()
        .expect("cancelled original console prefix");
    assert_eq!(
        prefix.completion(),
        runmat_execution::console::ConsoleCompletion::Interrupted
    );
    assert_eq!(prefix.records().len(), 1);
    assert_eq!(
        prefix.records()[0].text.as_ref(),
        "original before termination"
    );
    service
        .cancel(&Value::Future(future.into()), CancellationReason::User)
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
