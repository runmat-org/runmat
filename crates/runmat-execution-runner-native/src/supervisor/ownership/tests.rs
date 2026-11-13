use super::*;
use crate::supervisor::filesystem::{read_json, unix_millis};
use crate::supervisor::{
    DurableProgramAttempt, LocalJobState, LocalSupervisor, LocalSupervisorConfig,
};
use runmat_execution::JobId;
use runmat_execution_runner::{AttemptReport, DriverAction, DriverCommand};
use std::sync::Arc;

fn admitted(
    temp: &tempfile::TempDir,
) -> (
    Arc<LocalSupervisor>,
    LocalSupervisorConfig,
    JobId,
    AttemptRequest,
) {
    let config = crate::supervisor::tests::test_config(temp);
    let supervisor = LocalSupervisor::open(config.clone()).unwrap();
    let context =
        futures::executor::block_on(supervisor.execution_context(Some("program-key"))).unwrap();
    let submission = crate::supervisor::tests::original_program_submission(context);
    let store = futures::executor::block_on(supervisor.store.lock());
    let (record, _) = store
        .create_program(&submission, unix_millis(), Default::default())
        .unwrap();
    let mut scheduler = futures::executor::block_on(supervisor.scheduler.lock());
    scheduler.tick().unwrap();
    let request = scheduler
        .admit(record.handle.id, &submission.task)
        .unwrap()
        .into_iter()
        .find_map(|action| match action {
            DriverAction::Launch(request) => Some(request),
            _ => None,
        })
        .unwrap();
    store
        .assign_program(
            record.handle.id,
            &DurableProgramAttempt {
                schema_version: 1,
                request: request.clone(),
                config: config.execution.clone(),
                job_directory: store.job_dir(record.handle.id),
            },
        )
        .unwrap();
    scheduler
        .command(DriverCommand::BackendReport(
            runmat_execution_runner::port::BackendReport::for_request(
                &request,
                AttemptReport::Started,
            ),
        ))
        .unwrap();
    drop(scheduler);
    drop(store);
    (supervisor, config, record.handle.id, request)
}

#[test]
fn kernel_attempt_lease_is_exclusive_original_bound_and_keeps_its_inode() {
    use std::os::unix::fs::MetadataExt;
    let temp = tempfile::tempdir().unwrap();
    let (_supervisor, config, job, request) = admitted(&temp);
    let directory = config.paths.jobs.join(job.to_string());
    let lease = DurableAttemptLease::acquire(&directory, &request).unwrap();
    crate::supervisor::driver_files::write_driver_marker(
        &directory,
        std::process::id(),
        Some(request.id),
    )
    .unwrap();
    let inode = std::fs::metadata(path(&directory, &request)).unwrap().ino();
    assert!(is_owned(&directory, &request).unwrap());
    assert!(DurableAttemptLease::acquire(&directory, &request).is_err());
    let mut foreign = request.clone();
    foreign.worker_id = runmat_execution::identity::WorkerId::derive(&[b"foreign-worker"]);
    assert!(lease.validate(&directory, &foreign).is_err());
    assert!(lease.validate(temp.path(), &request).is_err());
    drop(lease);
    assert!(!is_owned(&directory, &request).unwrap());
    assert!(DurableAttemptLease::acquire(&directory, &request).is_err());
    assert_eq!(
        std::fs::metadata(path(&directory, &request)).unwrap().ino(),
        inode
    );
}

#[tokio::test]
async fn saved_live_pid_without_original_lease_cannot_keep_attempt_alive() {
    let temp = tempfile::tempdir().unwrap();
    let (supervisor, _config, job, request) = admitted(&temp);
    {
        let store = supervisor.store.lock().await;
        crate::supervisor::driver_files::write_driver_marker(
            &store.job_dir(job),
            std::process::id(),
            Some(request.id),
        )
        .unwrap();
        let mut record = store.record(job).unwrap();
        record.state = LocalJobState::Running;
        record.driver_process_id = Some(std::process::id());
        store.write_record(&mut record).unwrap();
    }
    let attachment = supervisor.attach(job, 0, 0).await.unwrap();
    assert_eq!(attachment.record.state, LocalJobState::Indeterminate);
    let history = attachment.execution.unwrap();
    history.validate().unwrap();
    assert_eq!(history.attempts[0].request, request);
    assert!(history.attempts[0].terminal.is_none());
    let unresolved = history.attempts[0].unverified_retirement.as_ref().unwrap();
    assert!(unresolved.resources.is_none());
    assert!(unresolved.execution.is_none());
    assert!(unresolved.retired_at_driver_millis.is_none());
    assert!(history.resource_aggregate(1).is_err());
}

#[tokio::test]
async fn escalation_routes_original_control_and_cannot_signal_a_foreign_saved_pid() {
    let temp = tempfile::tempdir().unwrap();
    let (supervisor, _config, job, request) = admitted(&temp);
    let mut command = runmat_process_host::HostCommand::new("/bin/sleep");
    command.arguments = vec!["60".into()];
    command.stdio = runmat_process_host::StdioPolicy::Null;
    let mut unrelated = command.spawn().await.unwrap();
    let mut store = supervisor.store.lock().await;
    let mut scheduler = supervisor.scheduler.lock().await;
    let directory = store.job_dir(job);
    let lease = DurableAttemptLease::acquire(&directory, &request).unwrap();
    let mut record = store.record(job).unwrap();
    record.state = LocalJobState::Running;
    record.driver_process_id = unrelated.id();
    store.write_record(&mut record).unwrap();
    let now_millis = scheduler.driver.snapshot().now_millis;
    scheduler
        .command(DriverCommand::CancelScope {
            scope_id: request.scope_id,
            reason: runmat_execution::CancellationReason::User,
            now_millis,
        })
        .unwrap();
    let mut foreign = request.clone();
    foreign.worker_id = runmat_execution::identity::WorkerId::derive(&[b"foreign-worker"]);
    assert!(supervisor
        .dispatch_program_actions(
            &mut store,
            &mut scheduler,
            vec![DriverAction::Terminate(foreign)]
        )
        .await
        .is_err());
    let marker = crate::supervisor::program_store::cancel_path(&directory, request.id);
    assert!(!marker.exists());
    supervisor
        .dispatch_program_actions(
            &mut store,
            &mut scheduler,
            vec![DriverAction::Terminate(request.clone())],
        )
        .await
        .unwrap();
    let cancellation =
        read_json::<crate::process_provider::cancellation::CancellationCommand>(&marker).unwrap();
    assert_eq!(cancellation.request, request);
    assert_eq!(
        crate::process_provider::cancellation::reason(&request, cancellation.cause).unwrap(),
        runmat_execution::CancellationReason::User
    );
    assert!(is_owned(&directory, &request).unwrap());
    assert!(unrelated.try_wait().unwrap().is_none());
    unrelated.terminate_tree().await.unwrap();
    drop(lease);
}

#[tokio::test]
async fn retention_cannot_unlink_an_owned_provider_inode_after_terminal_history() {
    let temp = tempfile::tempdir().unwrap();
    let (supervisor, _config, job, request) = admitted(&temp);
    let store = supervisor.store.lock().await;
    let mut scheduler = supervisor.scheduler.lock().await;
    let directory = store.job_dir(job);
    let lease = DurableAttemptLease::acquire(&directory, &request).unwrap();
    scheduler
        .command(DriverCommand::BackendReport(
            runmat_execution_runner::port::BackendReport::for_request(
                &request,
                AttemptReport::Failed {
                    kind: runmat_execution_runner::AttemptFailureKind::Execution,
                    message: "original failure".into(),
                },
            ),
        ))
        .unwrap();
    let mut record = store.record(job).unwrap();
    record.state = LocalJobState::Failed;
    record.submitted_unix_millis = 1;
    record.updated_unix_millis = 1;
    record.retain_until_unix_millis = 1;
    store.write_record(&mut record).unwrap();
    assert!(store
        .gc(unix_millis(), |job| scheduler
            .job_has_provider(&store, job)
            .map(|owned| !owned))
        .unwrap()
        .is_empty());
    assert!(directory.exists());
    drop(lease);
    assert_eq!(
        store
            .gc(unix_millis(), |job| scheduler
                .job_has_provider(&store, job)
                .map(|owned| !owned))
            .unwrap(),
        vec![job]
    );
    assert!(!directory.exists());
}
