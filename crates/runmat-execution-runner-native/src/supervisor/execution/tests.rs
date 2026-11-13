use crate::supervisor::{test_support, LocalSupervisor, LocalSupervisorConfig, SupervisorPaths};
use crate::NativeExecutionError;

#[tokio::test]
async fn unreadable_worker_artifact_rejects_before_job_and_scheduler_publication() {
    let temp = tempfile::tempdir().unwrap();
    let mut execution = crate::NativeExecutionConfig::for_current_executable().unwrap();
    execution.max_workers = 1;
    execution.max_stderr_bytes = 1024;
    execution.max_object_bytes = 1024 * 1024;
    execution.max_message_bytes = 1;
    let config = LocalSupervisorConfig::new(
        execution,
        SupervisorPaths::new(temp.path().join("state")).unwrap(),
    )
    .unwrap();
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context = supervisor
        .execution_context(Some("original-artifact-bound"))
        .await
        .unwrap();
    let journal = config.paths.root.join("scheduler.json");
    let original = std::fs::read(&journal).unwrap();
    let submission = test_support::original_program_submission(context);
    let result = supervisor.submit_program(submission).await;
    assert!(matches!(result, Err(NativeExecutionError::Encoding(error))
        if error.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit));
    assert!(supervisor.list().await.unwrap().is_empty());
    assert_eq!(std::fs::read(&journal).unwrap(), original);
    assert!(!config.paths.root.join("artifacts").exists());
    drop(supervisor);
    let reopened = LocalSupervisor::open(config).unwrap();
    assert_eq!(
        reopened
            .execution_context(Some("original-artifact-bound"))
            .await
            .unwrap(),
        context
    );
    assert!(reopened.list().await.unwrap().is_empty());
}
