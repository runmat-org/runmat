//! Failed cumulative cleanup admission cannot delete a prefix of an owned attempt.
use super::*;

#[test]
fn cleanup_retains_validated_paths_and_bindings_until_the_whole_attempt_is_admitted() {
    let _guard = crate::analysis::tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let identity = Digest::sha256(b"original-private-attempt");
    let attempt = directory(&store, identity);
    let records = attempt.join("runs");
    std::fs::create_dir_all(&records).unwrap();
    for ordinal in 0..256 {
        std::fs::write(records.join(format!("original_{ordinal:04}.json")), b"{}").unwrap();
    }
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    let entries = scope.fs_read_dir(&records).unwrap();
    let response_bytes: u64 = entries
        .iter()
        .map(|entry| entry.resident_bytes().unwrap())
        .sum();
    drop(entries);
    drop(scope);
    let scope = store
        .access_with_budget(true, response_bytes + 4096)
        .unwrap();
    assert_eq!(scope.fs_read_dir(&records).unwrap().len(), 256);
    assert!(matches!(
        uncommitted_roots(&scope, None).unwrap_err(),
        AnalysisPublicationError::ResourceLimit { .. }
    ));
    assert_eq!(std::fs::read_dir(&records).unwrap().count(), 256);
    drop(scope);
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    uncommitted_roots(&scope, None).unwrap();
    assert!(!attempt.exists());
}
