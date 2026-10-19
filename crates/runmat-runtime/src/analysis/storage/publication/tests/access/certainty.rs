//! Public commit must not attribute an earlier uncertain workspace root to a new study.
use super::*;
use runmat_filesystem::workspace::WorkspacePublicationReceipt;
use std::os::unix::fs::PermissionsExt;

struct Restore(PathBuf, std::fs::Permissions);
impl Drop for Restore {
    fn drop(&mut self) {
        std::fs::set_permissions(&self.0, self.1.clone()).unwrap();
    }
}

#[test]
fn cold_public_commit_preserves_prior_uncertainty_without_publishing_new_children() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let original = store
        .commit_publication(prepare(&run, &["original_closed_child"]))
        .unwrap();
    let journal = root.path().join(".artifacts/workspace-publications");
    let records: Vec<_> = std::fs::read_dir(journal.join("records"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(records.len(), 1);
    let directory = &records[0];
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("commit.json")).unwrap()).unwrap();
    let workspace: WorkspacePublicationReceipt =
        serde_json::from_value(record["receipt"].clone()).unwrap();
    // Re-expose the persisted recovery obligation, then obstruct a fresh directory
    // synchronization. No numerical solver or alternative reference is involved.
    std::fs::write(
        journal.join("pending.json"),
        serde_json::to_vec(&workspace.id).unwrap(),
    )
    .unwrap();
    let restore = Restore(
        directory.clone(),
        std::fs::metadata(directory).unwrap().permissions(),
    );
    std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o300)).unwrap();
    let incoming = prepare(&run, &["new_unpublished_child"]);
    let incoming_identity = incoming.descriptor.identity().unwrap();
    let expected = AnalysisPublicationError::DurabilityIndeterminate {
        workspace,
        publication: None,
    };
    assert_eq!(store.commit_publication(incoming).unwrap_err(), expected);
    assert!(!filesystem::directory(&store, incoming_identity).exists());
    drop(restore);
    let cold = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    assert!(cold.load_run("new_unpublished_child").unwrap().is_none());
    assert!(cold.load_run("original_closed_child").unwrap().is_some());
    assert!(cold
        .load_publication(original.identity, 1 << 20)
        .unwrap()
        .is_some());
    let resumed = cold
        .commit_publication(prepare(&run, &["new_unpublished_child"]))
        .unwrap();
    assert_eq!(resumed.identity, incoming_identity);
}
