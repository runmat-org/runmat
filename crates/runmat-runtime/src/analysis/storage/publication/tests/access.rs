use super::*;

#[cfg(unix)]
mod certainty;

#[test]
fn removed_root_and_binding_cannot_resurrect_a_cached_child_as_a_standalone_run() {
    let _guard = tests::analysis_test_guard();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let receipt = store
        .commit_publication(prepare(&run, &["cached_closed_child"]))
        .unwrap();
    assert!(store.load_run("cached_closed_child").unwrap().is_some());
    std::fs::remove_file(filesystem::directory(&store, receipt.identity).join("commit.json"))
        .unwrap();
    std::fs::remove_file(
        root.path()
            .join("publication-bindings/cached_closed_child.json"),
    )
    .unwrap();
    assert!(store.load_run("cached_closed_child").unwrap().is_none());
    assert!(store
        .load_run_metadata("cached_closed_child")
        .unwrap()
        .is_none());
    assert!(store.list_runs().unwrap().is_empty());
}

#[test]
fn captured_filesystem_provider_cannot_be_redirected_by_an_ambient_override() {
    let _guard = tests::analysis_test_guard();
    let _provider_guard = runmat_filesystem::provider_override_lock();
    reset_artifact_store_for_tests();
    let run = run();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let _override = runmat_filesystem::replace_provider(Arc::new(
        runmat_filesystem::MemoryFsProvider::with_current_dir("/"),
    ));
    let receipt = store
        .commit_publication(prepare(&run, &["captured_provider_child"]))
        .unwrap();
    assert!(store.load_run("captured_provider_child").unwrap().is_some());
    assert!(filesystem::directory(&store, receipt.identity)
        .join("commit.json")
        .is_file());
}
