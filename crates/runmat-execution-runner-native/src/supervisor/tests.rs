use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use runmat_execution::resource::ResourceRequest;
use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionResponse,
};
use runmat_execution_runner::AttemptResourceAccounting;

use super::auth::{constant_time_eq, read_token};
use super::model::{BatchSubmission, LocalJobState, ProgramBatchSubmission, MIN_RETENTION_MILLIS};
use super::store::SupervisorPaths;
use super::{LocalSupervisor, LocalSupervisorConfig};

mod borrowing;

pub(super) fn test_config(temp: &tempfile::TempDir) -> LocalSupervisorConfig {
    let executable = temp.path().join("driver.sh");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
set -eu
job="$RUNMAT_EXECUTION_JOB_DIR"
tmp="$job/driver.helper.tmp"
printf '{"schema_version":2,"process_id":%s,"attempt_id":null}' "$$" > "$tmp"
mv "$tmp" "$job/driver.json"
if grep -q LONG "$job"/*.m; then
  sleep 60
else
  : > "$job/helper.ready"
  while [ ! -f "$job/helper.release" ]; do sleep 0.01; done
fi
tmp="$job/completion.helper.tmp"
printf '{"schema_version":4,"success":true,"exit_code":0,"message":null}' > "$tmp"
mv "$tmp" "$job/completion.json"
: > "$job/helper.completed"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    execution.executable = executable;
    execution.max_workers = 1;
    execution.max_stderr_bytes = 1024;
    execution.max_object_bytes = 1024 * 1024;
    LocalSupervisorConfig::new(
        execution,
        SupervisorPaths::new(temp.path().join("state")).unwrap(),
    )
    .unwrap()
}

fn submission(source: &[u8], key: Option<&str>) -> BatchSubmission {
    BatchSubmission {
        source_name: "job.m".into(),
        source: source.to_vec(),
        arguments: Vec::new(),
        working_directory: std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        idempotency_key: key.map(str::to_owned),
        retention_millis: MIN_RETENTION_MILLIS,
    }
}

#[tokio::test]
async fn detached_job_survives_supervisor_reopen_and_submission_is_idempotent() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let first = LocalSupervisor::open(config.clone()).unwrap();
    let (record, duplicate) = first
        .submit(submission(b"pause(0.2)", Some("same-request")))
        .await
        .unwrap();
    assert!(!duplicate);
    let (same, duplicate) = first
        .submit(submission(b"pause(0.2)", Some("same-request")))
        .await
        .unwrap();
    assert!(duplicate);
    assert_eq!(same.handle.id, record.handle.id);
    assert!(first
        .submit(submission(b"different", Some("same-request")))
        .await
        .is_err());
    let job_dir = config.paths.jobs.join(record.handle.id.to_string());
    if !wait_for_helper(&job_dir.join("helper.ready")).await {
        let diagnostic = first.show(record.handle.id).await;
        let _ = first.cancel(record.handle.id).await;
        panic!("driver did not reach readiness handshake: {diagnostic:?}");
    }
    drop(first);

    let reopened = LocalSupervisor::open(config).unwrap();
    // The detached driver is demonstrably alive across reopen and cannot finish
    // until this supervisor releases it. Time passing is not completion evidence.
    let running = reopened.show(record.handle.id).await.unwrap();
    if running.state != LocalJobState::Running {
        let _ = reopened.cancel(record.handle.id).await;
        panic!("driver did not survive supervisor reopen: {running:?}");
    }
    std::fs::write(job_dir.join("helper.release"), b"release").unwrap();
    if !wait_for_helper(&job_dir.join("helper.completed")).await {
        let diagnostic = reopened.show(record.handle.id).await;
        let _ = reopened.cancel(record.handle.id).await;
        panic!("driver did not acknowledge durable completion: {diagnostic:?}");
    }
    let recovered = reopened.show(record.handle.id).await.unwrap();
    assert_eq!(recovered.state, LocalJobState::Succeeded);
    assert_eq!(recovered.exit_code, Some(0));
}

async fn wait_for_helper(marker: &std::path::Path) -> bool {
    // Bound deadlock diagnostics, not a presumed process-start/finish duration.
    // File markers avoid uncancellable blocking FIFO reads during test teardown.
    tokio::time::timeout(Duration::from_secs(30), async {
        while !marker.exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .is_ok()
}

#[tokio::test]
async fn cancellation_is_durable_and_corrupt_metadata_recovers_indeterminate() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let (record, _) = supervisor.submit(submission(b"LONG", None)).await.unwrap();
    let cancelled = supervisor.cancel(record.handle.id).await.unwrap();
    assert_eq!(cancelled.state, LocalJobState::Cancelled);

    let record_path = config
        .paths
        .jobs
        .join(record.handle.id.to_string())
        .join("record.json");
    std::fs::write(&record_path, b"{not-json").unwrap();
    let recovered = supervisor.show(record.handle.id).await.unwrap();
    assert_eq!(recovered.state, LocalJobState::Indeterminate);
    assert!(recovered
        .message
        .as_deref()
        .unwrap()
        .contains("metadata was corrupt"));
    assert_eq!(
        std::fs::metadata(&record_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(&config.paths.root)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    let mut expired = recovered;
    expired.submitted_unix_millis = 1;
    expired.updated_unix_millis = 1;
    expired.retain_until_unix_millis = 1;
    {
        let store = supervisor.store.lock().await;
        store.write_record(&expired).unwrap();
        assert_eq!(store.gc(2, |_| Ok(true)).unwrap(), vec![record.handle.id]);
    }
    assert!(!config
        .paths
        .jobs
        .join(record.handle.id.to_string())
        .exists());
}

#[test]
fn authentication_comparison_checks_length_and_content() {
    assert!(constant_time_eq(b"same", b"same"));
    assert!(!constant_time_eq(b"same", b"samf"));
    assert!(!constant_time_eq(b"same", b"same-longer"));
}

#[test]
fn token_reader_rejects_broad_permissions() {
    let temp = tempfile::tempdir().unwrap();
    let token = temp.path().join("token");
    std::fs::write(&token, "a".repeat(64)).unwrap();
    std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o644)).unwrap();
    let error = read_token(&token).unwrap_err();
    assert!(error.to_string().contains("mode-0600"));
}

#[test]
fn exact_program_submission_round_trips_through_durable_storage() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context =
        futures::executor::block_on(supervisor.execution_context(Some("program-key"))).unwrap();
    let store = futures::executor::block_on(supervisor.store.lock());
    let submission = original_program_submission(context);
    let (record, created) = store.create_program(&submission, 1).unwrap();
    assert!(created);
    let mut scheduler = futures::executor::block_on(supervisor.scheduler.lock());
    let request = scheduler
        .admit(record.handle.id, &submission.task)
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            runmat_execution_runner::DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    let assignment = super::DurableProgramAttempt {
        schema_version: 1,
        request: request.clone(),
        config: config.execution.clone(),
        job_directory: store.job_dir(record.handle.id),
    };
    store.assign_program(record.handle.id, &assignment).unwrap();
    assert_eq!(
        store.program_assignment(record.handle.id).unwrap(),
        Some(assignment)
    );
    let (duplicate, created) = store.create_program(&submission, 2).unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);
    let mut report = runmat_execution_runner::port::BackendReport::for_request(
        &request,
        runmat_execution_runner::AttemptReport::Failed {
            kind: runmat_execution_runner::AttemptFailureKind::Execution,
            message: "typed worker failure".into(),
        },
    );
    report.unsuccessful_resources = Some(Box::new(
        AttemptResourceAccounting::for_attempt(&request, 10, 0, 0, 0).unwrap(),
    ));
    let completed = super::ProgramBatchExecution {
        response: ProgramExecutionResponse::Failure {
            message: "typed worker failure".into(),
        },
        report,
        latest_progress: None,
    };
    super::program_store::write_program_completion(
        &store.job_dir(record.handle.id),
        completed.clone(),
    )
    .unwrap();
    assert_eq!(
        store.program_report(record.handle.id, &request).unwrap(),
        Some(completed.clone())
    );
    let mut substituted = completed.clone();
    substituted
        .report
        .unsuccessful_resources
        .as_mut()
        .unwrap()
        .resources
        .usage
        .cpu_core_millis += 1;
    assert!(super::program_store::write_program_completion(
        &store.job_dir(record.handle.id),
        substituted
    )
    .is_err());
    assert_eq!(
        store.program_report(record.handle.id, &request).unwrap(),
        Some(completed.clone())
    );
    scheduler
        .command(runmat_execution_runner::DriverCommand::BackendReport(
            completed.report,
        ))
        .unwrap();
    let original = scheduler.history(record.handle.id).unwrap().unwrap();
    original.validate().unwrap();
    let journal_before = std::fs::read(config.paths.root.join("scheduler.json")).unwrap();
    assert!(scheduler.tick_if_active().unwrap().is_empty());
    assert_eq!(
        std::fs::read(config.paths.root.join("scheduler.json")).unwrap(),
        journal_before
    );
    drop(scheduler);
    drop(store);
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    let scheduler = futures::executor::block_on(reopened.scheduler.lock());
    assert_eq!(
        scheduler.history(record.handle.id).unwrap().unwrap(),
        original
    );
}

fn program_resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1024 * 1024,
        scratch_bytes: 1024 * 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 1024 * 1024,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: Vec::new(),
        required_capabilities: Default::default(),
    }
}

#[test]
fn supervisor_open_requires_kernel_ownership_and_retains_one_lock_inode() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    let first = LocalSupervisor::open(config.clone()).unwrap();
    // Contents, including a stale or reused PID, cannot transfer live authority.
    std::fs::write(&config.paths.lock, b"0\n").unwrap();
    assert!(LocalSupervisor::open(config.clone()).is_err());
    drop(first);
    assert!(config.paths.lock.exists());
    let second = LocalSupervisor::open(config.clone()).unwrap();
    assert!(LocalSupervisor::open(config.clone()).is_err());
    drop(second);
    assert!(LocalSupervisor::open(config).is_ok());
}

pub(super) fn original_program_submission(
    context: super::DurableExecutionContext,
) -> ProgramBatchSubmission {
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "7".into(),
        outputs: OutputContract {
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
        b"exact-program".to_vec(),
    )
    .unwrap();
    let task = runmat_execution_runner::TaskSubmission {
        request: runmat_execution::task::TaskRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            id: runmat_execution::TaskId::derive(&[b"original-durable-program"]),
            scope_id: context.scope_id,
            pool_id: context.pool_id,
            program_artifact_id: runmat_execution::identity::ArtifactId::derive(&[artifact
                .id
                .0
                .bytes()]),
            callable: runmat_execution::task::Callable {
                owner_identity: "test".into(),
                qualified_name: "7".into(),
                entrypoint_digest: Digest::sha256(b"7"),
            },
            inputs: Vec::new(),
            outputs: OutputContract {
                requested_outputs: 1,
            },
            resources: program_resources(),
            retry: runmat_execution::task::RetryPolicy::Never,
            deadline_unix_millis: None,
        },
        dependencies: Default::default(),
        priority: 0,
    };
    ProgramBatchSubmission {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        recipe,
        artifact,
        function: 7,
        arguments: Vec::new(),
        requested_outputs: 1,
        task,
        idempotency_key: Some("program-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    }
}

#[tokio::test]
async fn failed_detached_launch_retains_original_unknown_attempt_accounting() {
    let temp = tempfile::tempdir().unwrap();
    let config = test_config(&temp);
    std::fs::remove_file(&config.execution.executable).unwrap();
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context = supervisor
        .execution_context(Some("program-key"))
        .await
        .unwrap();
    let submission = original_program_submission(context);
    let original_task = submission.task.clone();
    let (record, duplicate) = supervisor.submit_program(submission).await.unwrap();
    assert!(!duplicate);
    let attachment = supervisor.attach(record.handle.id, 0, 0).await.unwrap();
    assert!(attachment.record.state.is_terminal());
    let history = attachment.execution.unwrap();
    history.validate().unwrap();
    assert_eq!(history.task.submission, original_task);
    assert_eq!(history.attempts.len(), 1);
    let terminal = history.attempts[0].terminal.as_ref().unwrap();
    assert!(terminal.resources.is_none());
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    assert_eq!(
        reopened
            .attach(record.handle.id, 0, 0)
            .await
            .unwrap()
            .execution,
        Some(history)
    );
}

mod submission_wire;
