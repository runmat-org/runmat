//! Original native-protocol worker, lease, cancellation and cold report chain.
//! This fixture checks execution ownership, not FEA physics or measured RSS.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use runmat_execution::resource::{Capability, ResourceRequest};
use runmat_execution::{Digest, OutputContract, ProgramRevision};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};
use runmat_execution_runner::TaskSubmission;
use runmat_execution_runner_native::supervisor::{LocalSupervisorClient, ProgramBatchSubmission};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Ready {
    process_id: u32,
    artifact_id: runmat_execution_artifact::ProgramArtifactId,
    resources: ResourceRequest,
}

pub(super) async fn blocked_worker() {
    use runmat_process_host::ipc::{read_payload, FrameLimits};
    let mut control = runmat_process_host::ipc::private_control::connect_host()
        .await
        .expect("original attempt private control connection");
    let payload = read_payload(
        &mut control.reader,
        FrameLimits {
            max_message_bytes: 64 * 1024 * 1024,
        },
    )
    .await
    .unwrap();
    let reserved: runmat_execution_artifact::ReservedProgramExecutionRequest =
        serde_json::from_slice(&payload).unwrap();
    reserved.validate().unwrap();
    let (request, resources) = reserved.into_parts();
    let object_root =
        std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV).unwrap();
    let root = Path::new(&object_root).parent().unwrap();
    runmat_filesystem::NativeFsProvider
        .atomic_replace_sync(
            &root.join("worker-ready.json"),
            &serde_json::to_vec(&Ready {
                process_id: std::process::id(),
                artifact_id: request.artifact.id,
                resources,
            })
            .unwrap(),
        )
        .unwrap();
    // Deliberately block the worker's runtime. The original independent provider
    // must cancel and observe this actual process, without its cooperation.
    std::thread::sleep(Duration::from_secs(60));
}

struct CancelOriginalOnDrop {
    directory: PathBuf,
    request: runmat_execution_runner::AttemptRequest,
}
impl Drop for CancelOriginalOnDrop {
    fn drop(&mut self) {
        let cancel = self
            .directory
            .join(format!("{}.cancel.json", self.request.id));
        if let Ok(bytes) = serde_json::to_vec(&self.request) {
            let _ = runmat_filesystem::NativeFsProvider.atomic_replace_sync(&cancel, &bytes);
        }
        // Keep the fixture namespace available to the independent provider on
        // assertion failure, until its terminal publication or bounded teardown.
        let report = self
            .directory
            .join(format!("{}.completion.json", self.request.id));
        let lease = self
            .directory
            .join(format!("{}.provider.lock", self.request.id));
        let started = std::time::Instant::now();
        while started.elapsed() < Duration::from_secs(5) {
            if report.exists()
                && runmat_filesystem::ExclusiveFileLock::try_acquire(&lease)
                    .is_ok_and(|owner| owner.is_some())
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[derive(Clone, Copy)]
enum ControlCase {
    UserCancellation,
    QuotaHandle,
    ForeignAssignment,
}

pub(super) async fn run() {
    run_case(ControlCase::UserCancellation).await;
    run_case(ControlCase::QuotaHandle).await;
    run_case(ControlCase::ForeignAssignment).await;
}

async fn run_case(case: ControlCase) {
    let name = match case {
        ControlCase::UserCancellation => "controlled-authority",
        ControlCase::QuotaHandle => "quota-handle-authority",
        ControlCase::ForeignAssignment => "foreign-control-authority",
    };
    let stop = super::SupervisorFixture::new(name);
    let root = stop.root().to_path_buf();
    // Independently configured capacity for this new physical-process case;
    // the original small-inventory bootstrap reference stays unchanged.
    let mut execution =
        runmat_execution_runner_native::NativeExecutionConfig::for_current_executable().unwrap();
    execution.worker_arguments = vec!["--original-cohort-worker".into()];
    execution.max_workers = 1;
    execution.worker_memory_bytes = 512 * 1024 * 1024;
    execution.worker_scratch_bytes = 256 * 1024 * 1024;
    execution.worker_cpu_millicores = 1_000;
    let config = runmat_execution_runner_native::supervisor::LocalSupervisorConfig::new(
        execution,
        runmat_execution_runner_native::supervisor::SupervisorPaths::new(root.clone()).unwrap(),
    )
    .unwrap();
    let client = LocalSupervisorClient::new(config);
    client.connect_or_start().await.unwrap();
    let context = client
        .execution_context(Some("controlled-original-task"))
        .await
        .unwrap();
    let recipe = ProgramBuildRecipe {
        foreign_policy: None,
        accelerators: Vec::new(),
        interop: runmat_types::InteropManifest::empty(),
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: ProgramRevision::new(
            Digest::sha256(b"control-graph"),
            Digest::sha256(b"control-source"),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap(),
        entrypoint: "7".into(),
        outputs: OutputContract {
            owned_work: None,
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: runmat_execution_artifact::ProgramTarget::portable("test-interpreter-bytecode-v1"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV1,
        b"controlled protocol worker".to_vec(),
    )
    .unwrap();
    let task = TaskSubmission {
        request: runmat_execution::task::TaskRequest {
            schema_version: runmat_execution::task::TASK_REQUEST_SCHEMA_VERSION,
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: runmat_execution::TaskId::derive(&[b"controlled-original-task"]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            executable: runmat_execution::task::TaskExecutableIdentity::ProgramArtifact(
                runmat_execution::identity::ArtifactId::derive(&[artifact.id.0.bytes()]),
            ),
            callable: runmat_execution::task::Callable::for_program(
                "test",
                &runmat_execution::ProgramCallable::semantic(
                    runmat_execution::ProgramFunctionId(7),
                    None,
                ),
            ),
            host: artifact
                .execution_host_requirement(
                    &recipe,
                    BTreeSet::from([
                        runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
                    ]),
                )
                .unwrap(),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            inputs: Vec::new(),
            outputs: recipe.outputs.clone(),
            resources: ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 256 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024,
                max_wall_millis: 10_000,
                max_artifact_bytes: 1024 * 1024,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    let original_artifact = artifact.id;
    let (record, disposition) = client
        .submit_program(ProgramBatchSubmission {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            recipe,
            artifact,
            callable: runmat_execution::ProgramCallable::semantic(
                runmat_execution::ProgramFunctionId(7),
                None,
            ),
            invocation_context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: Vec::new(),
            requested_outputs: 1,
            task: task.clone(),
            idempotency_key: Some("controlled-original-task".into()),
            retention_millis: runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
        })
        .await
        .unwrap();
    assert_eq!(
        disposition,
        runmat_execution_runner_native::supervisor::JobSubmissionDisposition::Created
    );
    let directory = root.join("jobs").join(record.handle.id.to_string());
    let assignment: runmat_execution_runner_native::supervisor::DurableProgramAttempt =
        serde_json::from_slice(
            &runmat_filesystem::NativeFsProvider
                .read_bounded_sync(&directory.join("assignment.json"), 64 * 1024)
                .unwrap(),
        )
        .unwrap();
    let cleanup = CancelOriginalOnDrop {
        directory: directory.clone(),
        request: assignment.request,
    };
    let ready = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let path = root.join("worker-ready.json");
            if path.exists() {
                break serde_json::from_slice::<Ready>(
                    &runmat_filesystem::NativeFsProvider
                        .read_bounded_sync(&path, 1024)
                        .unwrap(),
                )
                .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original native worker did not reach the control handshake");
    assert_eq!(ready.artifact_id, original_artifact);
    assert_eq!(ready.resources, task.request.resources);
    let running = client
        .attach(record.handle.id, 0, 0)
        .await
        .unwrap()
        .execution
        .unwrap();
    assert_eq!(running.task.submission, task);
    assert_eq!(running.attempts.len(), 1);
    let request = &running.attempts[0].request;
    let lease_path = directory.join(format!("{}.provider.lock", request.id));
    assert!(
        runmat_filesystem::ExclusiveFileLock::try_acquire(&lease_path)
            .unwrap()
            .is_none()
    );
    match case {
        ControlCase::UserCancellation => {
            client.cancel(record.handle.id).await.unwrap();
        }
        ControlCase::QuotaHandle => {
            for mutation in 0..3 {
                let mut altered = record.handle.clone();
                match mutation {
                    0 => altered.outputs.requested_outputs += 1,
                    1 => altered.generation += 1,
                    _ => altered.run_id = runmat_execution::RunId::derive(&[b"foreign-run"]),
                }
                assert!(client
                    .cancel_handle(altered, runmat_execution::CancellationReason::Quota)
                    .await
                    .is_err());
                let retained = client
                    .attach(record.handle.id, 0, 0)
                    .await
                    .unwrap()
                    .execution
                    .unwrap();
                assert_eq!(retained.task.cancellation, None);
                assert_eq!(retained.attempts[0].request, *request);
                assert!(runmat_process_host::is_process_alive(ready.process_id));
            }
            client
                .cancel_handle(
                    record.handle.clone(),
                    runmat_execution::CancellationReason::Quota,
                )
                .await
                .unwrap();
        }
        ControlCase::ForeignAssignment => {
            let mut foreign = request.clone();
            foreign.worker_id =
                runmat_execution::identity::WorkerId::derive(&[b"foreign-control-worker"]);
            runmat_filesystem::NativeFsProvider
                .atomic_replace_sync(
                    &directory.join(format!("{}.cancel.json", request.id)),
                    &serde_json::to_vec(&foreign).unwrap(),
                )
                .unwrap();
        }
    }
    let closed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let attachment = client.attach(record.handle.id, 0, 0).await.unwrap();
            if attachment.record.state.is_terminal() {
                break attachment;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("original native worker cancellation did not close");
    let expected_state = match case {
        ControlCase::UserCancellation | ControlCase::QuotaHandle => {
            runmat_execution_runner_native::supervisor::LocalJobState::Cancelled
        }
        ControlCase::ForeignAssignment => {
            runmat_execution_runner_native::supervisor::LocalJobState::Failed
        }
    };
    assert_eq!(closed.record.state, expected_state);
    let original = closed.execution.unwrap();
    original.validate().unwrap();
    assert_eq!(original.task.submission, task);
    assert_eq!(original.attempts.len(), 1);
    assert_eq!(&original.attempts[0].request, request);
    if matches!(
        case,
        ControlCase::UserCancellation | ControlCase::QuotaHandle
    ) {
        let reason = if matches!(case, ControlCase::QuotaHandle) {
            runmat_execution::CancellationReason::Quota
        } else {
            runmat_execution::CancellationReason::User
        };
        assert_eq!(
            original.task.cancellation.unwrap().cause,
            runmat_execution_runner::TaskCancellationCause::Task {
                task_id: request.task_id,
                reason
            }
        );
        client
            .cancel_handle(
                record.handle.clone(),
                runmat_execution::CancellationReason::User,
            )
            .await
            .unwrap();
        assert_eq!(
            client
                .attach(record.handle.id, 0, 0)
                .await
                .unwrap()
                .execution
                .unwrap(),
            original
        );
    }
    if matches!(case, ControlCase::ForeignAssignment) {
        assert!(matches!(
            original.attempts[0].terminal.as_ref().unwrap().disposition,
            runmat_execution_runner::AttemptDisposition::Failed {
                kind: runmat_execution_runner::AttemptFailureKind::Execution,
            }
        ));
    }
    let observation = original.attempts[0]
        .terminal
        .as_ref()
        .unwrap()
        .execution
        .as_ref()
        .unwrap();
    assert!(matches!(
        observation.termination,
        runmat_execution_runner::AttemptExecutionTermination::ProcessExited { .. }
    ));
    assert!(
        runmat_filesystem::ExclusiveFileLock::try_acquire(&lease_path)
            .unwrap()
            .is_some()
    );
    assert!(!runmat_process_host::is_process_alive(ready.process_id));

    // Terminate the actual supervisor fixture, then bootstrap the same frozen
    // profile under new kernel ownership and compare the entire original history.
    std::fs::write(root.join("test-stop"), b"stop").unwrap();
    super::wait_stopped(&client, &root).await;
    std::fs::remove_file(root.join("test-stop")).unwrap();
    std::fs::remove_file(root.join("test-stopped")).unwrap();
    client.connect_or_start().await.unwrap();
    assert_eq!(
        client
            .attach(record.handle.id, 0, 0)
            .await
            .unwrap()
            .execution
            .unwrap(),
        original
    );
    drop(cleanup);
    stop.request_stop().unwrap();
    super::wait_stopped(&client, &root).await;
    drop(stop);
    let label = match case {
        ControlCase::UserCancellation => "original user cancellation",
        ControlCase::QuotaHandle => "original quota/complete-handle cancellation",
        ControlCase::ForeignAssignment => "foreign-control failure",
    };
    println!("{label}, physical exit and cold original history conformance passed");
}
