use super::*;
use crate::analysis::{configuration, tests::analysis_test_guard};

struct Restore(crate::analysis::FeaRuntimeConfig);
impl Drop for Restore {
    fn drop(&mut self) {
        crate::analysis::configure_fea_runtime(self.0.clone()).unwrap();
    }
}

#[test]
fn carried_inventory_budget_admits_native_list_and_read_without_global_redirection() {
    let _guard = analysis_test_guard();
    let previous = configuration::current_fea_runtime_config();
    let _restore = Restore(previous.clone());
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("original"), b"original content").unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access_with_budget(false, 4096).unwrap();
    let mut changed = previous;
    changed.publication_staging_bytes = 1;
    crate::analysis::configure_fea_runtime(changed).unwrap();
    let entries = scope.fs_read_dir(root.path()).unwrap();
    let original = entries
        .iter()
        .find(|entry| entry.path() == root.path().join("original"))
        .unwrap();
    assert_eq!(original.file_type(), runmat_filesystem::FsFileType::File);
    assert_eq!(scope.fs_read(original.path()).unwrap(), b"original content");
    drop(scope);
    let limited = store.access(false).unwrap();
    assert!(limited.fs_read_dir(root.path()).is_err());
    assert!(limited.fs_read(root.path().join("original")).is_err());
}

#[test]
fn sparse_oversized_original_rejects_before_content_buffer_is_read() {
    let _guard = analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let file = std::fs::File::create(root.path().join("oversized")).unwrap();
    file.set_len(1 << 30).unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access_with_budget(false, 4096).unwrap();
    let entries = scope.fs_read_dir(root.path()).unwrap();
    let original = entries
        .iter()
        .find(|entry| entry.path() == root.path().join("oversized"))
        .unwrap();
    assert!(scope.fs_read(original.path()).is_err());
    assert_eq!(std::fs::metadata(original.path()).unwrap().len(), 1 << 30);
}

#[test]
fn cold_metadata_node_density_is_admitted_before_original_schema_decoding() {
    let _guard = analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access_with_budget(false, 80 * 1024).unwrap();
    let dense = serde_json::to_vec(&vec![0; 1000]).unwrap();
    assert!(dense.len() < 80 * 1024);
    let error = scope.decode_record(&dense, "original").err().unwrap();
    assert!(matches!(
        error,
        AnalysisPublicationError::ResourceLimit { .. }
    ));
    assert!(scope
        .admit_json_metadata(br#"{"original":"content"}"#)
        .is_ok());
}

#[test]
fn coexisting_directory_and_content_share_one_ceiling_and_release_on_error() {
    let _guard = analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("original");
    std::fs::write(&path, vec![42; 1024]).unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access_with_budget(false, 1024).unwrap();
    assert_eq!(scope.fs_read(&path).unwrap().len(), 1024);
    let entries = scope.fs_read_dir(root.path()).unwrap();
    let reservation = scope.retain_directory(&entries).unwrap();
    assert!(scope.remaining_bytes() < 1024);
    assert!(scope.fs_read(&path).is_err());
    assert!(scope.reserve_resident(1024).is_err());
    drop(entries);
    drop(reservation);
    assert_eq!(scope.fs_read(&path).unwrap().len(), 1024);
    assert_eq!(scope.remaining_bytes(), 1024);
}

#[test]
fn cold_decoding_obeys_nested_reservations_and_non_lifo_release() {
    let _guard = analysis_test_guard();
    let root = tempfile::tempdir().unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root.path().to_path_buf());
    let scope = store.access_with_budget(false, 128 * 1024).unwrap();
    let bytes = br#"{"original":"content"}"#;
    let first = scope.reserve_resident(32 * 1024).unwrap();
    assert!(scope.admit_json_metadata(bytes).is_ok());
    let mut second = scope.reserve_resident(32 * 1024).unwrap();
    let before = scope.remaining_bytes();
    assert!(second.grow(u64::MAX).is_err());
    assert_eq!(scope.remaining_bytes(), before);
    assert!(scope.admit_json_metadata(bytes).is_err());
    drop(first);
    assert!(scope.admit_json_metadata(bytes).is_ok());
    drop(second);
    assert_eq!(scope.remaining_bytes(), 128 * 1024);
}
