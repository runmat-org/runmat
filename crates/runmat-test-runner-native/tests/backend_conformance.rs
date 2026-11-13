use runmat_test::identity::RunId;
use runmat_test_runner::artifact::{persist_reports, ArtifactStore};
use runmat_test_runner::reporter::RenderedReport;
use runmat_test_runner_native::artifact::{export_reports, FilesystemArtifactStore};

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

    let unrelated = temporary.path().join("user-notes.txt");
    std::fs::write(&unrelated, b"keep me").unwrap();

    store.remove_run(&run_id).await.unwrap();
    assert!(!temporary
        .path()
        .join(&manifest.artifacts[0].store_key)
        .exists());
    assert_eq!(std::fs::read(unrelated).unwrap(), b"keep me");
}

#[tokio::test]
async fn direct_report_export_replaces_only_validated_requested_names() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let unrelated = root.join("user-notes.txt");
    std::fs::write(&unrelated, b"keep me").unwrap();

    let mut reports = vec![RenderedReport {
        name: "test-results.json".into(),
        media_type: "application/json".into(),
        bytes: b"first".to_vec(),
    }];
    export_reports(root, &reports).await.unwrap();
    assert_eq!(
        std::fs::read(root.join("test-results.json")).unwrap(),
        b"first"
    );

    reports[0].bytes = b"second".to_vec();
    export_reports(root, &reports).await.unwrap();
    assert_eq!(
        std::fs::read(root.join("test-results.json")).unwrap(),
        b"second"
    );
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep me");

    for name in [
        "",
        ".",
        "..",
        "nested/results.json",
        "..\\results.json",
        "bad\0name",
        "CON.txt",
        "lpt9.xml",
        "report:stream",
        "report.",
        "report ",
        "réport.json",
    ] {
        reports[0].name = name.into();
        assert!(export_reports(root, &reports).await.is_err(), "{name:?}");
    }
    reports[0].name = "test-results.json".into();
    let mut duplicate = reports[0].clone();
    duplicate.name = "TEST-RESULTS.JSON".into();
    assert!(export_reports(root, &[reports[0].clone(), duplicate])
        .await
        .is_err());
    let fresh = RenderedReport {
        name: "coverage.json".into(),
        media_type: "application/json".into(),
        bytes: b"fresh".to_vec(),
    };
    let invalid = RenderedReport {
        name: "../escape.json".into(),
        media_type: "application/json".into(),
        bytes: b"invalid".to_vec(),
    };
    assert!(export_reports(root, &[fresh, invalid]).await.is_err());
    assert!(!root.join("coverage.json").exists());
    assert_eq!(
        std::fs::read(root.join("test-results.json")).unwrap(),
        b"second"
    );
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep me");

    #[cfg(unix)]
    {
        let external = root.join("external.txt");
        std::fs::write(&external, b"external").unwrap();
        std::fs::remove_file(root.join("test-results.json")).unwrap();
        std::os::unix::fs::symlink(&external, root.join("test-results.json")).unwrap();
        export_reports(root, &reports).await.unwrap();
        assert_eq!(std::fs::read(external).unwrap(), b"external");
        assert_eq!(
            std::fs::read(root.join("test-results.json")).unwrap(),
            b"second"
        );
    }
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
