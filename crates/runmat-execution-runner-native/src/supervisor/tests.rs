use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{ExecutableForm, ProgramArtifact, ProgramBuildRecipe};

use super::auth::{constant_time_eq, read_token};
use super::model::{
    BatchDriverInvocation, BatchSubmission, LocalJobState, ProgramBatchSubmission,
    MIN_RETENTION_MILLIS,
};
use super::service::{LocalSupervisor, LocalSupervisorConfig};
use super::store::{load_driver_invocation, JobStore, SupervisorPaths};

fn test_config(temp: &tempfile::TempDir) -> LocalSupervisorConfig {
    let executable = temp.path().join("driver.sh");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
set -eu
job="$RUNMAT_EXECUTION_JOB_DIR"
tmp="$job/driver.helper.tmp"
printf '{"schema_version":1,"process_id":%s}' "$$" > "$tmp"
mv "$tmp" "$job/driver.json"
if grep -q LONG "$job"/*.m; then
  sleep 60
else
  sleep 0.2
fi
tmp="$job/completion.helper.tmp"
printf '{"schema_version":1,"success":true,"exit_code":0,"message":null,"value":null}' > "$tmp"
mv "$tmp" "$job/completion.json"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    LocalSupervisorConfig {
        executable,
        paths: SupervisorPaths::new(temp.path().join("state")).unwrap(),
        max_stderr_bytes: 1024,
    }
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
    drop(first);

    let reopened = LocalSupervisor::open(config).unwrap();
    let mut recovered = reopened.show(record.handle.id).await.unwrap();
    for _ in 0..50 {
        if recovered.state.is_terminal() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        recovered = reopened.show(record.handle.id).await.unwrap();
    }
    assert_eq!(recovered.state, LocalJobState::Succeeded);
    assert_eq!(recovered.exit_code, Some(0));
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
        assert_eq!(store.gc(2).unwrap(), vec![record.handle.id]);
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
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
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
    let submission = ProgramBatchSubmission {
        recipe,
        artifact,
        function: 7,
        arguments: Vec::new(),
        requested_outputs: 1,
        idempotency_key: Some("program-key".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let (record, created) = store.create_program(submission.clone(), 1).unwrap();
    assert!(created);
    let invocation = load_driver_invocation(&store.job_dir(record.handle.id)).unwrap();
    let BatchDriverInvocation::Program {
        submission: recovered,
        ..
    } = invocation
    else {
        panic!("expected an exact program invocation");
    };
    assert_eq!(recovered.function, submission.function);
    assert_eq!(recovered.artifact, submission.artifact);
    let (duplicate, created) = store.create_program(submission, 2).unwrap();
    assert!(!created);
    assert_eq!(duplicate.handle, record.handle);
}
