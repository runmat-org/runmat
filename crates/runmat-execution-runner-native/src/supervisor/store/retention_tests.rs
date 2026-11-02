use super::{BatchSubmission, JobStore, LocalJobState, SupervisorPaths, MIN_RETENTION_MILLIS};

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
