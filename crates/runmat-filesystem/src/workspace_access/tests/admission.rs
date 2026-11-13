use super::*;

#[test]
fn scoped_commit_admits_future_recovery_before_allocating_a_durable_record() {
    use crate::workspace::{WorkspaceEdit, WorkspacePublicationId};
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source");
    std::fs::write(&path, b"original").unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 128 << 20)
        .unwrap();
    let snapshot = scope
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: Vec::new(),
                max_files: 10,
                max_bytes: 1024,
            },
            1 << 20,
        )
        .unwrap();
    let bytes = vec![7; 128 * 1024];
    let request = WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: Vec::new(),
        expected_revision: snapshot.revision,
        edits: vec![WorkspaceEdit::Write {
            path: "source".into(),
            bytes: bytes.clone(),
        }],
    };
    assert!(scope.publish(request.clone(), 1 << 20).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    assert!(!root
        .path()
        .join(".artifacts/workspace-publications/pending.json")
        .exists());
    assert!(!root
        .path()
        .join(".artifacts/workspace-publications/records")
        .exists());
    assert!(matches!(
        scope.publish(request, 128 << 20).unwrap(),
        WorkspacePublicationOutcome::Published { .. }
    ));
    assert_eq!(scope.read(Path::new("source"), 1 << 20).unwrap(), bytes);
}

#[test]
fn scoped_capture_rejects_sparse_content_under_the_carried_resident_ceiling() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("oversized");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(1 << 30)
        .unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), false, 1 << 20)
        .unwrap();
    assert!(scope
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: Vec::new(),
                max_files: 10,
                max_bytes: 1 << 30,
            },
            1 << 20
        )
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 1 << 30);
}

#[test]
fn insufficient_publication_phase_admission_preserves_source_and_allows_exact_retry() {
    use crate::workspace::{WorkspaceEdit, WorkspacePublicationId};
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("source");
    std::fs::write(&path, b"original").unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    let snapshot = scope
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: Vec::new(),
                max_files: 10,
                max_bytes: 1024,
            },
            1 << 20,
        )
        .unwrap();
    let request = WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: Vec::new(),
        expected_revision: snapshot.revision,
        edits: vec![WorkspaceEdit::Write {
            path: "source".into(),
            bytes: b"replacement".to_vec(),
        }],
    };
    assert!(scope.publish(request.clone(), 200_000).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    assert!(!root
        .path()
        .join(".artifacts/workspace-publications/records")
        .exists());
    let outcome = scope.publish(request.clone(), 1 << 20).unwrap();
    assert!(matches!(
        outcome,
        WorkspacePublicationOutcome::Published { .. }
    ));
    assert_eq!(std::fs::read(&path).unwrap(), b"replacement");
    assert_eq!(scope.publish(request, 1 << 20).unwrap(), outcome);
}

#[test]
fn immutable_collision_reads_reject_sparse_oversize_and_preserve_original_content() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("original");
    std::fs::File::create(&path)
        .unwrap()
        .set_len(1 << 30)
        .unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    assert!(scope
        .atomic_create_new(Path::new("original"), b"replacement", 16)
        .is_err());
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 1 << 30);
    std::fs::write(&path, b"original").unwrap();
    assert!(scope
        .atomic_create_new(Path::new("original"), b"replacement", 7)
        .is_err());
    assert!(
        matches!(scope.atomic_create_new(Path::new("original"), b"replacement", 8).unwrap(),
        AtomicCreateResult::Existing(bytes) if bytes == b"original")
    );
    assert!(std::fs::read_dir(root.path()).unwrap().all(|entry| !entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".publish-")));
}
