use runmat_test::identity::RunId;
use runmat_test_runner::artifact::{persist_reports, ArtifactStore};
use runmat_test_runner::reporter::RenderedReport;
use runmat_test_runner_native::artifact::FilesystemArtifactStore;

#[tokio::test]
async fn filesystem_store_persists_safe_reports_and_removes_a_run() {
    let temporary = tempfile::tempdir().unwrap();
    let store = FilesystemArtifactStore::new(temporary.path());
    let run_id = RunId::derive("revision", "native-artifacts");
    let reports = vec![RenderedReport {
        name: "reports/results.json".into(),
        media_type: "application/json".into(),
        bytes: br#"{"ok":true}"#.to_vec(),
    }];

    let manifest = persist_reports(&store, &run_id, &reports).await.unwrap();
    assert_eq!(manifest.artifacts.len(), 1);
    assert_eq!(
        manifest.artifacts[0].byte_len,
        reports[0].bytes.len() as u64
    );
    assert!(temporary
        .path()
        .join(&manifest.artifacts[0].store_key)
        .is_file());

    store.remove_run(&run_id).await.unwrap();
    assert!(!temporary
        .path()
        .join(&manifest.artifacts[0].store_key)
        .exists());
}

#[tokio::test]
async fn native_cancellation_is_sticky_and_first_reason_wins() {
    use runmat_test_runner::host::CancellationPort;
    use runmat_test_runner_native::host::NativeCancellation;

    let cancellation = NativeCancellation::default();
    cancellation.cancel("first");
    cancellation.cancel("second");
    assert!(cancellation.is_cancelled());
    assert_eq!(cancellation.reason().as_deref(), Some("first"));
    assert_eq!(cancellation.cancelled().await, "first");
}
