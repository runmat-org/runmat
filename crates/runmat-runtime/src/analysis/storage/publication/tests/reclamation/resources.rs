//! A bounded directory response is insufficient admission for concurrent sweep inventories.
use super::*;

#[test]
fn sweep_inventory_rejection_preserves_every_orphan_then_larger_scope_collects() {
    let _guard = tests::analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    std::fs::create_dir(store.objects_directory()).unwrap();
    for ordinal in 0u32..256 {
        let bytes = ordinal.to_le_bytes();
        std::fs::write(store.object_path(Digest::sha256(bytes)), bytes).unwrap();
    }
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    let entries = scope.fs_read_dir(store.objects_directory()).unwrap();
    let directory_bytes: u64 = entries
        .iter()
        .map(|entry| entry.resident_bytes().unwrap())
        .sum();
    drop(entries);
    drop(scope);
    let scope = store
        .access_with_budget(true, directory_bytes + 4096)
        .unwrap();
    // The original provider response itself fits this same captured ceiling.
    assert_eq!(
        scope.fs_read_dir(store.objects_directory()).unwrap().len(),
        256
    );
    let error = filesystem::reclamation::sweep(&scope, std::iter::empty()).unwrap_err();
    assert!(matches!(
        error,
        AnalysisPublicationError::ResourceLimit { .. }
    ));
    assert_eq!(
        std::fs::read_dir(store.objects_directory())
            .unwrap()
            .count(),
        256
    );
    drop(scope);
    let scope = store.access_with_budget(true, 16 << 20).unwrap();
    assert_eq!(
        filesystem::reclamation::sweep(&scope, std::iter::empty()).unwrap(),
        filesystem::reclamation::SweepOutcome::Reclaimed
    );
    assert_eq!(
        std::fs::read_dir(store.objects_directory())
            .unwrap()
            .count(),
        0
    );
}
