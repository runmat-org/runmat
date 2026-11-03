//! Actual native host deadlines without an analysis coordinator or remote scheduler.
use super::*;
#[path = "attempt_lifecycle/spawn_failure.rs"]
mod spawn_failure;
use runmat_execution_artifact::ReservedProgramExecutionRequest;
use runmat_execution_runner::{AttemptDisposition, AttemptExecutionTermination};
use runmat_execution_runner_native::WorkerProcessMessage;
use runmat_process_host::ipc::{read_payload, write_payload, FrameLimits};
use std::path::PathBuf;

pub(super) async fn fault_child(mode: &str) {
    let (mut reader, mut writer) = runmat_process_host::ipc::stdio::endpoint().unwrap();
    let limits = FrameLimits {
        max_message_bytes: 64 * 1024 * 1024,
    };
    let bytes = read_payload(&mut reader, limits).await.unwrap();
    let reserved: ReservedProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
    reserved.validate().unwrap();
    drop(reserved);
    let mut response = ProgramExecutionResponse::Success {
        outputs: vec![ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::Null,
        ))],
    };
    if mode == "--retry-observed-child" {
        // Test-only fault injection; no production formulation or physical output.
        let root = std::env::var_os(NATIVE_OBJECT_STORE_ROOT_ENV).unwrap();
        match std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(Path::new(&root).join("first-attempt-witness"))
        {
            Ok(_) => {
                response = ProgramExecutionResponse::Failure {
                    message: "intentional first attempt failure".into(),
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("fault injection marker: {error}"),
        }
    }
    if mode == "--checkpoint-failure-child" {
        // Test-only storage fault in this fresh session. Execution still succeeds.
        let objects = PathBuf::from(std::env::var_os(NATIVE_OBJECT_STORE_ROOT_ENV).unwrap());
        let checkpoints = objects.parent().unwrap().join("checkpoints");
        std::fs::remove_dir_all(&checkpoints).unwrap();
        std::fs::write(checkpoints, b"injected checkpoint namespace failure").unwrap();
    }
    let bytes = if mode == "--raw-terminal-child" {
        serde_json::to_vec(&response).unwrap()
    } else {
        serde_json::to_vec(&WorkerProcessMessage::Completed {
            response,
            console: None,
            console_scopes: Vec::new(),
        })
        .unwrap()
    };
    write_payload(&mut writer, &bytes, limits).await.unwrap();
    if mode == "--exit-after-success-child" {
        std::process::exit(17);
    }
}

async fn fault_parent() {
    for mode in [
        "--exit-after-success-child",
        "--raw-terminal-child",
        "--retry-observed-child",
        "--checkpoint-failure-child",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let (host, request) = fixture();
        let session = NativeProgramSession::new(config(directory.path(), mode)).unwrap();
        seed_host_invocation(&mut session.object_store(), &host, &request);
        let mut original = submission(&session, &host, &request);
        if mode == "--retry-observed-child" {
            original.request.retry =
                runmat_execution::task::RetryPolicy::TestPolicy { max_attempts: 2 };
        }
        let envelope = original.retry_resource_envelope().unwrap();
        let task = session.submit(request, original.clone()).unwrap();
        let result = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Some(result) = task.try_result() {
                    break result;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let history = task.try_attempt_history().unwrap();
        let evidence = task.try_execution_evidence().unwrap();
        evidence.validate().unwrap();
        assert_eq!(evidence.task.submission, original);
        assert_eq!(evidence.attempts, history);
        assert_eq!(
            serde_json::from_slice::<runmat_execution_runner::TaskExecutionEvidence>(
                &serde_json::to_vec(&evidence).unwrap()
            )
            .unwrap(),
            evidence
        );
        assert!(history.len() <= usize::from(envelope.maximum_attempts));
        assert_eq!(
            history.len(),
            if mode == "--retry-observed-child" {
                2
            } else {
                1
            },
            "{mode}: {result:?}"
        );
        assert_eq!(result.is_ok(), mode == "--retry-observed-child");
        for (index, attempt) in history.iter().enumerate() {
            assert_eq!(attempt.request.ordinal, u16::try_from(index + 1).unwrap());
            let terminal = attempt.terminal.as_ref().unwrap();
            let observation = terminal.execution.as_ref().unwrap();
            observation.validate_against(&attempt.request).unwrap();
            assert!(matches!(
                observation.termination,
                AttemptExecutionTermination::ProcessExited { .. }
            ));
            assert!(observation.local_ipc_sent_bytes.unwrap() > 0);
            assert!(observation.local_ipc_received_bytes.unwrap() > 0);
        }
        if mode == "--exit-after-success-child" {
            assert_eq!(
                history[0]
                    .terminal
                    .as_ref()
                    .unwrap()
                    .execution
                    .as_ref()
                    .unwrap()
                    .termination,
                AttemptExecutionTermination::ProcessExited {
                    code: Some(17),
                    success: false
                }
            );
        }
        if mode == "--checkpoint-failure-child" {
            assert!(matches!(
                task.try_checkpoint_evidence(),
                Some(runmat_execution_runner_native::NativeCheckpointEvidence::NotPublished { .. })
            ));
            assert!(task.try_scheduler_result().unwrap().is_ok());
            assert_eq!(
                history[0].terminal.as_ref().unwrap().disposition,
                AttemptDisposition::Succeeded
            );
            assert!(result
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("checkpoint was not published"));
        } else {
            assert_eq!(
                task.try_checkpoint_evidence(),
                Some(runmat_execution_runner_native::NativeCheckpointEvidence::Durable)
            );
        }
        if mode == "--retry-observed-child" {
            assert!(history[0].terminal.as_ref().unwrap().resources.is_none());
            assert_eq!(
                history[0].terminal.as_ref().unwrap().diagnostic.as_deref(),
                Some("intentional first attempt failure")
            );
            let success = result.unwrap();
            assert_eq!(
                history[1].terminal.as_ref().unwrap().resources.as_deref(),
                success.resources()
            );
            assert_eq!(
                history[1]
                    .terminal
                    .as_ref()
                    .unwrap()
                    .execution
                    .as_ref()
                    .unwrap()
                    .wall_millis,
                success.resources().unwrap().resources.usage.wall_millis
            );
        }
        let cold: Vec<runmat_execution_runner::task::AttemptRecord> =
            serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
        assert_eq!(cold, history);
        eprintln!("native lifecycle fault {mode}: original history/exit/checkpoint evidence PASS");
    }
}

pub(super) async fn run() {
    for user_cancel in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let (host, request) = fixture();
        let session = NativeProgramSession::new(config(directory.path(), "--slow-child")).unwrap();
        seed_host_invocation(&mut session.object_store(), &host, &request);
        let mut task_submission = submission(&session, &host, &request);
        if !user_cancel {
            task_submission.request.resources.max_wall_millis = 250;
        }
        let task = session.submit(request, task_submission).unwrap();
        if user_cancel {
            tokio::time::timeout(Duration::from_secs(5), async {
                while task.drain_progress().is_empty() {
                    assert!(task.try_result().is_none());
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            session.cancel(runmat_execution::CancellationReason::User);
        }
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(result) = task.try_result() {
                    break result;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(matches!(
            result,
            Err(runmat_execution_runner_native::NativeProgramFailure::Cancelled)
        ));
        let history = task.try_attempt_history().unwrap();
        assert_eq!(history.len(), 1);
        let original = &history[0];
        let terminal = original.terminal.as_ref().unwrap();
        assert_eq!(terminal.disposition, AttemptDisposition::Cancelled);
        assert!(terminal.closed_at_driver_millis >= original.assigned_at_millis);
        assert!(original.cancellation_requested_at.is_some());
        // Original process interval/transport/exit evidence is available, but
        // complete artifact accounting must not be invented from reservation.
        assert!(terminal.resources.is_none());
        let execution = terminal.execution.as_ref().unwrap();
        assert!(execution.wall_millis > 0);
        assert!(matches!(
            execution.termination,
            AttemptExecutionTermination::ProcessExited { .. }
        ));
        assert!(execution.local_ipc_sent_bytes.unwrap() > 0);
        if user_cancel {
            assert!(execution.local_ipc_received_bytes.unwrap() > 0);
        }
        if !user_cancel {
            assert!(
                original.cancellation_requested_at.unwrap()
                    >= original.assigned_at_millis
                        + original.request.task.resources.max_wall_millis
            );
        }
        let cold: Vec<runmat_execution_runner::task::AttemptRecord> =
            serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
        assert_eq!(cold, history);
    }
    fault_parent().await;
    spawn_failure::run().await;
}
