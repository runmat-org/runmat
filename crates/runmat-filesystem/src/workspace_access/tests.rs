use super::*;

mod admission;
use crate::{FsProvider, NativeFsProvider, SandboxFsProvider};
use std::{sync::mpsc, time::Duration};

#[cfg(any(unix, windows))]
#[test]
fn immutable_evidence_synchronizes_relative_and_absolute_workspace_roots() {
    // Keep process-wide cwd unchanged: both names identify this owned directory.
    let directory = tempfile::tempdir_in(".").unwrap();
    let relative = std::path::PathBuf::from(directory.path().file_name().unwrap());
    let absolute = std::fs::canonicalize(directory.path()).unwrap();
    for root in [absolute, relative] {
        let scope = NativeFsProvider
            .workspace_access(&root, true, 1 << 20)
            .unwrap();
        scope.create_dir_all(Path::new("nested")).unwrap();
        scope
            .atomic_create_new(Path::new("nested/evidence"), b"original", 1 << 20)
            .unwrap();
        scope
            .synchronize_immutable(Path::new("nested/evidence"))
            .unwrap();
        assert_eq!(
            scope.read(Path::new("nested/evidence"), 1 << 20).unwrap(),
            b"original"
        );
        assert_eq!(
            scope
                .synchronize_immutable(Path::new("nested/absent"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }
}

#[test]
fn the_same_access_scope_recovers_a_durable_partial_batch_before_reading() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("a.fea"), b"old-a").unwrap();
    std::fs::write(root.path().join("b.fea"), b"old-b").unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    let outcome = crate::workspace_native::failed_commit_for_test(root.path());
    assert!(matches!(
        outcome,
        WorkspacePublicationOutcome::CommittedPendingRecovery { .. }
    ));
    assert!(scope.read(Path::new("a.fea"), 1 << 20).is_err());
    std::fs::remove_dir(root.path().join("b.fea")).unwrap();
    assert_eq!(scope.read(Path::new("a.fea"), 1 << 20).unwrap(), b"new-a");
    assert_eq!(scope.read(Path::new("b.fea"), 1 << 20).unwrap(), b"new-b");
}

#[cfg(unix)]
#[test]
fn a_replaced_root_cannot_redirect_an_existing_scope() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("root");
    std::fs::create_dir(&root).unwrap();
    let scope = NativeFsProvider
        .workspace_access(&root, true, 1 << 20)
        .unwrap();
    scope
        .atomic_create_new(Path::new("record"), b"original", 1 << 20)
        .unwrap();
    let original = parent.path().join("original-root");
    std::fs::rename(&root, &original).unwrap();
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("record"), b"replacement").unwrap();
    assert_eq!(
        scope.read(Path::new("record"), 1 << 20).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(
        scope
            .atomic_create_new(Path::new("new-record"), b"bad", 1 << 20)
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(!root.join("new-record").exists());
    assert_eq!(std::fs::read(original.join("record")).unwrap(), b"original");
}

#[cfg(windows)]
#[test]
fn windows_scope_retains_the_mount_until_physical_retirement() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("root");
    let original = parent.path().join("original-root");
    std::fs::create_dir(&root).unwrap();
    let scope = NativeFsProvider
        .workspace_access(&root, true, 1 << 20)
        .unwrap();
    scope
        .atomic_create_new(Path::new("record"), b"original", 1 << 20)
        .unwrap();
    // Windows retains the namespace through its opened directory/journal
    // handles; unlike Unix, replacement is refused before it can redirect IO.
    assert!(std::fs::rename(&root, &original).is_err());
    assert!(!original.exists());
    assert_eq!(
        scope.read(Path::new("record"), 1 << 20).unwrap(),
        b"original"
    );
    drop(scope);
    // Prove that the refusal was tied to the live owner, not a permanently
    // inaccessible fixture. Retirement releases the physical namespace.
    std::fs::rename(&root, &original).unwrap();
    assert!(!root.exists());
    assert_eq!(std::fs::read(original.join("record")).unwrap(), b"original");
}

#[test]
fn competing_threads_in_one_scope_have_one_conditional_batch_winner() {
    use crate::workspace::{WorkspaceEdit, WorkspacePublicationId};
    let root = tempfile::tempdir().unwrap();
    let scope = std::sync::Arc::new(
        NativeFsProvider
            .workspace_access(root.path(), true, 1 << 20)
            .unwrap(),
    );
    let snapshot = scope
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: vec![],
                max_files: 32,
                max_bytes: 1024,
            },
            1 << 20,
        )
        .unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let mut handles = Vec::new();
    for content in [b"one", b"two"] {
        let scope = scope.clone();
        let start = barrier.clone();
        let request = WorkspacePublicationRequest {
            id: WorkspacePublicationId::new(),
            root: "/".into(),
            excluded_subtrees: snapshot.excluded_subtrees.clone(),
            expected_revision: snapshot.revision,
            edits: vec![WorkspaceEdit::Write {
                path: "record".into(),
                bytes: content.to_vec(),
            }],
        };
        handles.push(std::thread::spawn(move || {
            start.wait();
            scope.publish(request, 1 << 20).unwrap()
        }));
    }
    let outcomes = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, WorkspacePublicationOutcome::Published { .. }))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|o| matches!(o, WorkspacePublicationOutcome::Conflict { .. }))
            .count(),
        1
    );
    assert!([b"one", b"two"].contains(
        &scope
            .read(Path::new("record"), 1 << 20)
            .unwrap()
            .as_slice()
            .try_into()
            .unwrap()
    ));
}

#[test]
fn scoped_batch_reuses_fence_and_has_a_cold_idempotent_receipt() {
    use crate::workspace::{WorkspaceEdit, WorkspacePublicationId};
    let root = tempfile::tempdir().unwrap();
    let writer = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    writer
        .atomic_create_new(Path::new("old/commit"), b"old", 1 << 20)
        .unwrap();
    let snapshot = writer
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: vec![],
                max_files: 32,
                max_bytes: 1024,
            },
            1 << 20,
        )
        .unwrap();
    let request = WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: snapshot.excluded_subtrees,
        expected_revision: snapshot.revision,
        edits: vec![
            WorkspaceEdit::Remove {
                path: "old/commit".into(),
            },
            WorkspaceEdit::Write {
                path: "new/commit".into(),
                bytes: b"new".to_vec(),
            },
        ],
    };
    let outcome = writer.publish(request.clone(), 1 << 20).unwrap();
    assert!(matches!(
        outcome,
        WorkspacePublicationOutcome::Published { .. }
    ));
    assert_eq!(
        writer.read(Path::new("new/commit"), 1 << 20).unwrap(),
        b"new"
    );
    assert_eq!(
        writer
            .read(Path::new("old/commit"), 1 << 20)
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    drop(writer);
    let reopened = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    assert_eq!(reopened.publish(request.clone(), 1 << 20).unwrap(), outcome);
    let mut substituted = request;
    substituted.edits.push(WorkspaceEdit::Write {
        path: "foreign".into(),
        bytes: vec![1],
    });
    assert_eq!(
        reopened.publish(substituted, 1 << 20).unwrap(),
        WorkspacePublicationOutcome::IdReused
    );
    assert!(reopened.read(Path::new("foreign"), 1 << 20).is_err());
    drop(reopened);
    let reader = NativeFsProvider
        .workspace_access(root.path(), false, 1 << 20)
        .unwrap();
    assert_eq!(
        reader.read(Path::new("new/commit"), 1 << 20).unwrap(),
        b"new"
    );
}

#[test]
fn scoped_batch_rejects_stale_inventory_and_read_only_access() {
    use crate::workspace::{WorkspaceEdit, WorkspacePublicationId};
    let root = tempfile::tempdir().unwrap();
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    let snapshot = scope
        .capture(
            WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: vec![],
                max_files: 32,
                max_bytes: 1024,
            },
            1 << 20,
        )
        .unwrap();
    scope
        .atomic_create_new(Path::new("later"), b"original", 1 << 20)
        .unwrap();
    let request = WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: snapshot.excluded_subtrees,
        expected_revision: snapshot.revision,
        edits: vec![WorkspaceEdit::Write {
            path: "uncommitted".into(),
            bytes: vec![1],
        }],
    };
    assert!(matches!(
        scope.publish(request.clone(), 1 << 20).unwrap(),
        WorkspacePublicationOutcome::Conflict { .. }
    ));
    assert!(scope.read(Path::new("uncommitted"), 1 << 20).is_err());
    drop(scope);
    let reader = NativeFsProvider
        .workspace_access(root.path(), false, 1 << 20)
        .unwrap();
    assert_eq!(
        reader.publish(request, 1 << 20).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
}

#[test]
fn competing_access_scopes_cannot_read_a_partial_writer_inventory() {
    let root = tempfile::tempdir().unwrap();
    let provider = NativeFsProvider;
    let writer = provider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    writer
        .atomic_create_new(Path::new("first"), b"one", 1 << 20)
        .unwrap();
    let path = root.path().to_path_buf();
    let (started_tx, started_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let scope = NativeFsProvider
            .workspace_access(&path, false, 1 << 20)
            .unwrap();
        let pair = (
            scope.read(Path::new("first"), 1 << 20).unwrap(),
            scope.read(Path::new("second"), 1 << 20).unwrap(),
        );
        finished_tx.send(pair).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(matches!(
        finished_rx.recv_timeout(Duration::from_millis(50)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    writer
        .atomic_create_new(Path::new("second"), b"two", 1 << 20)
        .unwrap();
    drop(writer);
    assert_eq!(
        finished_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        (b"one".to_vec(), b"two".to_vec())
    );
    reader.join().unwrap();
}

#[test]
fn absent_read_scope_stays_absent_and_cannot_write_or_escape() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("absent");
    let reader = NativeFsProvider
        .workspace_access(&root, false, 1 << 20)
        .unwrap();
    let writer = NativeFsProvider
        .workspace_access(&root, true, 1 << 20)
        .unwrap();
    writer
        .atomic_create_new(Path::new("record"), b"value", 1 << 20)
        .unwrap();
    assert_eq!(
        reader
            .read(Path::new("record"), 1 << 20)
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    assert_eq!(
        reader
            .atomic_create_new(Path::new("other"), b"bad", 1 << 20)
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(writer.read(Path::new("../escape"), 1 << 20).is_err());
    assert!(writer.read(Path::new("/absolute"), 1 << 20).is_err());
    assert!(writer
        .atomic_create_new(
            Path::new(".artifacts/workspace-publications/forged"),
            b"bad",
            1 << 20
        )
        .is_err());
}

#[test]
fn sandbox_scope_preserves_its_mount_and_avoids_nested_lock_reentry() {
    let root = tempfile::tempdir().unwrap();
    let provider = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    {
        let writer = provider
            .workspace_access(Path::new("/artifacts"), true, 1 << 20)
            .unwrap();
        writer
            .atomic_create_new(Path::new("record"), b"original", 1 << 20)
            .unwrap();
    }
    let reader = provider
        .workspace_access(Path::new("/artifacts"), false, 1 << 20)
        .unwrap();
    assert_eq!(
        reader.read(Path::new("record"), 1 << 20).unwrap(),
        b"original"
    );
    assert_eq!(
        std::fs::read(root.path().join("artifacts/record")).unwrap(),
        b"original"
    );
    assert!(provider
        .workspace_access(Path::new("../outside"), false, 1 << 20)
        .is_err());
}

#[test]
fn unsupported_provider_rejects_instead_of_opening_native_files() {
    let provider = crate::MemoryFsProvider::with_current_dir("/");
    assert!(
        matches!(provider.workspace_access(Path::new("/artifact"), true, 1 << 20), Err(error) if error.kind() == io::ErrorKind::Unsupported)
    );
}

#[cfg(unix)]
#[test]
fn symlink_roots_and_record_paths_cannot_redirect_scoped_access() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("redirect")).unwrap();
    assert!(NativeFsProvider
        .workspace_access(&root.path().join("redirect"), true, 1 << 20)
        .is_err());
    let scope = NativeFsProvider
        .workspace_access(root.path(), true, 1 << 20)
        .unwrap();
    assert!(scope.read(Path::new("redirect/record"), 1 << 20).is_err());
    assert!(scope
        .atomic_create_new(Path::new("redirect/record"), b"bad", 1 << 20)
        .is_err());
    assert!(!outside.path().join("record").exists());
}
