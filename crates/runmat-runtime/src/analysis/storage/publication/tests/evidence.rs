//! Actual captured-scope writes and hostile collision rejection, preserving original artifacts.
use super::*;

#[test]
fn evidence_capacity_rejects_before_creation_and_captures_original_root_and_provider() {
    let _guard = tests::analysis_test_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let root = tempfile::tempdir().unwrap();
    let first = root.path().join("first");
    let second = root.path().join("second");
    let _config = tests::scoped_study_artifact_root(&first);
    let definition = Digest::sha256(b"captured-evidence-scope");
    let denied = AnalysisPublicationBuilder::new(definition, 1).unwrap();
    assert!(matches!(
        denied.persist_evidence(
            &definition.to_string(),
            "original",
            serde_json::json!({"value": 17})
        ),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert!(!first.exists());
    let admitted = AnalysisPublicationBuilder::new(definition, 16 << 20).unwrap();
    let _changed_config = tests::scoped_study_artifact_root(&second);
    let _changed_provider =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::MemoryFsProvider::new()));
    let path = admitted
        .persist_evidence(
            &definition.to_string(),
            "original",
            serde_json::json!({"value": 17}),
        )
        .unwrap();
    assert!(PathBuf::from(&path).starts_with(&first));
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(original["value"], 17);
    assert!(!second.exists());
    assert_eq!(
        admitted
            .persist_evidence(
                &definition.to_string(),
                "original",
                serde_json::json!({"value": 17})
            )
            .unwrap(),
        path
    );
    assert!(admitted.store.list_runs().unwrap().is_empty());
}

#[test]
fn oversized_existing_evidence_rejects_bounded_read_without_replacing_original() {
    let _guard = tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let _config = tests::scoped_study_artifact_root(root.path());
    let definition = Digest::sha256(b"bounded-evidence-collision");
    let builder = AnalysisPublicationBuilder::new(definition, 16 << 20).unwrap();
    let path = builder
        .persist_evidence(
            &definition.to_string(),
            "collision",
            serde_json::json!({"original": true}),
        )
        .unwrap();
    let original = std::fs::read(&path).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(1 << 30).unwrap();
    assert!(matches!(
        builder.persist_evidence(
            &definition.to_string(),
            "collision",
            serde_json::json!({"original": true})
        ),
        Err(AnalysisPublicationError::ResourceLimit { .. })
    ));
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 1 << 30);
    // Rejection preserves the original prefix and does not poison a restored exact retry.
    use std::io::Read;
    let mut prefix = vec![0; original.len()];
    std::fs::File::open(&path)
        .unwrap()
        .read_exact(&mut prefix)
        .unwrap();
    assert_eq!(prefix, original);
    file.set_len(original.len() as u64).unwrap();
    assert_eq!(
        builder
            .persist_evidence(
                &definition.to_string(),
                "collision",
                serde_json::json!({"original": true})
            )
            .unwrap(),
        path
    );
    assert!(builder.store.list_runs().unwrap().is_empty());
}

#[cfg(unix)]
#[test]
fn exact_evidence_retry_does_not_launder_failed_directory_synchronization() {
    use std::os::unix::fs::PermissionsExt;
    let _guard = tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let _config = tests::scoped_study_artifact_root(root.path());
    let definition = Digest::sha256(b"evidence-sync-retry");
    let builder = AnalysisPublicationBuilder::new(definition, 16 << 20).unwrap();
    let payload = serde_json::json!({"original": [1, 2, 3]});
    let path = builder
        .persist_evidence(&definition.to_string(), "sync", payload.clone())
        .unwrap();
    let original = std::fs::read(&path).unwrap();
    let directory = PathBuf::from(&path).parent().unwrap().to_path_buf();
    struct Restore(PathBuf, std::fs::Permissions);
    impl Drop for Restore {
        fn drop(&mut self) {
            std::fs::set_permissions(&self.0, self.1.clone()).unwrap();
        }
    }
    let restore = Restore(
        directory.clone(),
        std::fs::metadata(&directory).unwrap().permissions(),
    );
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o300)).unwrap();
    let error = builder
        .persist_evidence(&definition.to_string(), "sync", payload.clone())
        .unwrap_err();
    assert!(
        matches!(error, AnalysisPublicationError::EvidenceDurabilityIndeterminate { path: ref retained, .. } if retained == &PathBuf::from(&path))
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(builder.store.list_runs().unwrap().is_empty());
    drop(restore);
    assert_eq!(
        builder
            .persist_evidence(&definition.to_string(), "sync", payload)
            .unwrap(),
        path
    );
}
