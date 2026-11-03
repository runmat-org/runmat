use super::{BatchSubmission, JobStore, LocalJobState, SupervisorPaths, MIN_RETENTION_MILLIS};
use runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization::{
    Default as DefaultOrg, Invalid, Selected,
};

#[test]
fn organization_is_per_job_durable_control_and_changes_idempotent_request_identity() {
    let temp = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(temp.path().join("state")).unwrap();
    let store = JobStore::open(paths.clone()).unwrap();
    let submission = BatchSubmission {
        source_name: "plain.m".into(),
        source: b"disp(1);".to_vec(),
        arguments: Vec::new(),
        working_directory: temp.path().to_string_lossy().into_owned(),
        idempotency_key: Some("same-call".into()),
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let org = uuid::Uuid::from_u128(1);
    let (record, created) = store
        .create_script(submission.clone(), 1, Selected(org))
        .unwrap();
    assert!(created);
    assert_eq!(record.local_fea_organization, Selected(org));
    assert_eq!(store.record(record.handle.id).unwrap(), record);
    assert!(
        !store
            .create_script(submission.clone(), 2, Selected(org))
            .unwrap()
            .1
    );
    assert!(store
        .create_script(submission.clone(), 3, Selected(uuid::Uuid::from_u128(2)))
        .is_err());
    assert!(store
        .create_script(submission.clone(), 4, DefaultOrg)
        .is_err());
    assert!(store.create_script(submission, 5, Invalid).is_err());
    drop(store);
    let reopened = JobStore::open(paths).unwrap();
    assert_eq!(
        reopened
            .record(record.handle.id)
            .unwrap()
            .local_fea_organization,
        Selected(org)
    );
    let plain = BatchSubmission {
        source_name: "still-plain.m".into(),
        source: b"disp(2);".to_vec(),
        arguments: Vec::new(),
        working_directory: temp.path().to_string_lossy().into_owned(),
        idempotency_key: None,
        retention_millis: MIN_RETENTION_MILLIS,
    };
    let (plain, created) = reopened.create_script(plain, 6, Invalid).unwrap();
    assert!(
        created,
        "an invalid FEA selector must not deny a plain deferred job"
    );
    assert_eq!(plain.local_fea_organization, Invalid);
}

#[test]
fn long_running_jobs_receive_terminal_retention_without_read_renewal() {
    let temp = tempfile::tempdir().unwrap();
    let store = JobStore::open(SupervisorPaths::new(temp.path().join("state")).unwrap()).unwrap();
    for terminal in [
        LocalJobState::Succeeded,
        LocalJobState::Failed,
        LocalJobState::Cancelled,
        LocalJobState::Indeterminate,
    ] {
        let submitted = 100;
        let (mut record, _) = store
            .create_script(
                BatchSubmission {
                    source_name: "original.m".into(),
                    source: b"original".to_vec(),
                    arguments: Vec::new(),
                    working_directory: temp.path().to_string_lossy().into_owned(),
                    idempotency_key: None,
                    retention_millis: MIN_RETENTION_MILLIS,
                },
                submitted,
                DefaultOrg,
            )
            .unwrap();
        let original_expiry = submitted + MIN_RETENTION_MILLIS;
        let terminal_at = original_expiry + 10;
        for active in [
            LocalJobState::Queued,
            LocalJobState::Starting,
            LocalJobState::Running,
            LocalJobState::Cancelling,
        ] {
            record.state = active;
            record.driver_process_id =
                matches!(active, LocalJobState::Running | LocalJobState::Cancelling)
                    .then_some(std::process::id());
            record.updated_unix_millis = terminal_at;
            store.write_record(&mut record).unwrap();
            assert_eq!(store.record(record.handle.id).unwrap(), record);
            assert!(store.gc(terminal_at, |_| Ok(true)).unwrap().is_empty());
        }
        record.state = terminal;
        record.driver_process_id = None;
        store.write_record(&mut record).unwrap();
        let expiry = terminal_at + MIN_RETENTION_MILLIS;
        assert_eq!(record.retain_until_unix_millis, expiry);
        assert_eq!(store.record(record.handle.id).unwrap(), record);
        store.write_record(&mut record).unwrap();
        assert_eq!(record.retain_until_unix_millis, expiry);
        assert!(store.gc(expiry - 1, |_| Ok(true)).unwrap().is_empty());
        assert_eq!(
            store.gc(expiry, |_| Ok(true)).unwrap(),
            vec![record.handle.id]
        );
    }
}
