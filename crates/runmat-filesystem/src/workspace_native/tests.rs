use super::*;
use crate::{workspace::*, FsProvider, OpenFlags, SandboxFsProvider};
use futures::executor::block_on;
use std::io::Write;

fn setup() -> (tempfile::TempDir, SandboxFsProvider) {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("a.fea"), b"old-a").unwrap();
    fs::write(root.path().join("b.fea"), b"old-b").unwrap();
    let provider = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    (root, provider)
}

fn request(provider: &SandboxFsProvider) -> WorkspacePublicationRequest {
    let snapshot = block_on(provider.capture_workspace(WorkspaceCaptureRequest {
        root: "/".into(),
        excluded_subtrees: vec![".artifacts".into()],
        max_files: 100,
        max_bytes: 1024,
    }))
    .unwrap();
    WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: snapshot.root,
        excluded_subtrees: snapshot.excluded_subtrees,
        expected_revision: snapshot.revision,
        edits: vec![
            WorkspaceEdit::Write {
                path: "a.fea".into(),
                bytes: b"new-a".to_vec(),
            },
            WorkspaceEdit::Write {
                path: "nested/b.fea".into(),
                bytes: b"new-b".to_vec(),
            },
            WorkspaceEdit::Remove {
                path: "b.fea".into(),
            },
        ],
    }
}

#[test]
fn native_publication_receipt_and_retry_survive_restart_and_later_edits() {
    let (root, provider) = setup();
    let input = request(&provider);
    assert_eq!(
        block_on(provider.lookup_workspace_publication(&input)).unwrap(),
        None
    );
    let result = block_on(provider.publish_workspace(input.clone())).unwrap();
    let WorkspacePublicationOutcome::Published { receipt } = &result else {
        panic!("not published")
    };
    let current = block_on(provider.capture_workspace(WorkspaceCaptureRequest {
        root: "/".into(),
        excluded_subtrees: input.excluded_subtrees.clone(),
        max_files: 100,
        max_bytes: 1024,
    }))
    .unwrap();
    assert_eq!(current.revision, receipt.revision);
    assert_eq!(
        block_on(provider.read(Path::new("nested/b.fea"))).unwrap(),
        b"new-b"
    );
    assert!(!root.path().join("b.fea").exists());
    assert!(!journal::record_directory(root.path(), input.id)
        .join("plan.json")
        .exists());
    drop(provider);
    let restored = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    block_on(restored.write(Path::new("a.fea"), b"later human draft")).unwrap();
    assert_eq!(
        block_on(restored.lookup_workspace_publication(&input)).unwrap(),
        Some(result.clone())
    );
    assert_eq!(
        block_on(restored.publish_workspace(input.clone())).unwrap(),
        result
    );
    assert_eq!(
        block_on(restored.read(Path::new("a.fea"))).unwrap(),
        b"later human draft"
    );
    let mut changed = input;
    changed.edits.clear();
    changed.edits.push(WorkspaceEdit::Write {
        path: "a.fea".into(),
        bytes: b"other".to_vec(),
    });
    assert_eq!(
        block_on(restored.lookup_workspace_publication(&changed)).unwrap(),
        Some(WorkspacePublicationOutcome::IdReused)
    );
    assert_eq!(
        block_on(restored.publish_workspace(changed)).unwrap(),
        WorkspacePublicationOutcome::IdReused
    );
}

#[test]
fn invalid_late_edit_and_stale_inventory_leave_all_source_unchanged() {
    let (root, provider) = setup();
    let original = request(&provider);
    let mut invalid = original.clone();
    invalid.edits.push(WorkspaceEdit::Remove {
        path: "missing.fea".into(),
    });
    assert!(block_on(provider.publish_workspace(invalid)).is_err());
    assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"old-a");
    assert!(!root.path().join("nested").exists());
    block_on(provider.write(Path::new("new_consumer.fea"), b"consumer")).unwrap();
    assert!(matches!(
        block_on(provider.publish_workspace(original)).unwrap(),
        WorkspacePublicationOutcome::Conflict { .. }
    ));
    assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"old-a");
}

#[test]
fn read_only_dependencies_and_reserved_journal_cannot_be_overwritten() {
    let (root, provider) = setup();
    let input = request(&provider);
    block_on(provider.set_readonly(Path::new("a.fea"), true)).unwrap();
    assert!(matches!(
        block_on(provider.publish_workspace(input)).unwrap(),
        WorkspacePublicationOutcome::Conflict { .. }
    ));
    let mut forbidden = request(&provider);
    forbidden.edits = vec![WorkspaceEdit::Write {
        path: "a.fea".into(),
        bytes: b"bad".to_vec(),
    }];
    assert_eq!(
        block_on(provider.publish_workspace(forbidden))
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(
        block_on(provider.write(Path::new(JOURNAL).join("pending.json").as_path(), b"forged"))
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert_eq!(
        block_on(provider.remove_dir_all(Path::new(".artifacts")))
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    block_on(provider.set_readonly(Path::new("a.fea"), false)).unwrap();
    assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"old-a");
}

fn commit_without_materialization(root: &Path, id: WorkspacePublicationId) -> PathBuf {
    let directory = journal::record_directory(root, id);
    fs::create_dir_all(directory.parent().unwrap()).unwrap();
    let plan = vec![
        journal::FinalEdit::Write {
            path: "a.fea".into(),
            bytes: b"new-a".to_vec(),
            mode: None,
        },
        journal::FinalEdit::Write {
            path: "b.fea".into(),
            bytes: b"new-b".to_vec(),
            mode: None,
        },
    ];
    journal::write_record(
        &directory,
        &plan,
        &mut journal::Record {
            digest: [1; 32],
            plan_digest: [0; 32],
            receipt: WorkspacePublicationReceipt {
                id,
                revision: WorkspaceRevision([2; 32]),
            },
        },
    )
    .unwrap();
    directory
}

/// Caller holds the shared journal fence. A real obstructed rename stops the
/// second materialization after the first write, without a fault-injection API.
pub(crate) fn failed_commit_for_test(root: &Path) -> WorkspacePublicationOutcome {
    let directory = commit_without_materialization(root, WorkspacePublicationId::new());
    fs::remove_file(root.join("b.fea")).unwrap();
    fs::create_dir(root.join("b.fea")).unwrap();
    let receipt = journal::read_record(&directory).unwrap().unwrap().receipt;
    let plan = vec![
        journal::FinalEdit::Write {
            path: "a.fea".into(),
            bytes: b"new-a".to_vec(),
            mode: None,
        },
        journal::FinalEdit::Write {
            path: "b.fea".into(),
            bytes: b"new-b".to_vec(),
            mode: None,
        },
    ];
    journal::finish_committed(root, &directory, &plan, receipt)
}

#[test]
fn materialization_failure_retains_a_typed_durable_receipt_and_cold_recovery() {
    let (root, provider) = setup();
    let outcome = {
        let _guard = guard(root.path(), true).unwrap();
        failed_commit_for_test(root.path())
    };
    let WorkspacePublicationOutcome::CommittedPendingRecovery { receipt } = &outcome else {
        panic!("durable commit misclassified")
    };
    let decoded: WorkspacePublicationOutcome =
        serde_json::from_slice(&serde_json::to_vec(&outcome).unwrap()).unwrap();
    assert_eq!(decoded, outcome);
    // a is physically new, but participating access must reject the incomplete batch.
    assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"new-a");
    assert!(block_on(provider.read(Path::new("a.fea"))).is_err());
    fs::remove_dir(root.path().join("b.fea")).unwrap();
    assert_eq!(
        block_on(provider.read(Path::new("b.fea"))).unwrap(),
        b"new-b"
    );
    let directory = journal::record_directory(root.path(), receipt.id);
    assert_eq!(
        journal::read_record(&directory).unwrap().unwrap().receipt,
        *receipt
    );
    assert!(directory.join("complete").exists());
    assert!(!directory.join("plan.json").exists());
}

#[test]
fn rejected_commit_fence_discards_preparation_without_publishing_source() {
    let (root, provider) = setup();
    let requested = request(&provider);
    let directory = journal::record_directory(root.path(), requested.id);
    let mut record = journal::Record {
        digest: publication_digest(&requested).unwrap(),
        plan_digest: [0; 32],
        receipt: WorkspacePublicationReceipt {
            id: requested.id,
            revision: WorkspaceRevision([2; 32]),
        },
    };
    {
        let _guard = guard(root.path(), true).unwrap();
        fs::create_dir_all(directory.parent().unwrap()).unwrap();
        let result = journal::write_record_checked(
            &directory,
            &[journal::FinalEdit::Write {
                path: "a.fea".into(),
                bytes: b"unpublished".to_vec(),
                mode: None,
            }],
            &mut record,
            || {
                assert!(directory.join("plan.json").exists());
                Err(io::Error::other(crate::execution::LostLease))
            },
        );
        assert!(result
            .unwrap_err()
            .get_ref()
            .unwrap()
            .is::<crate::execution::LostLease>());
        assert!(!directory.join("commit.json").exists());
        assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"old-a");
    }
    // Participating reads discard the uncommitted journal before returning.
    assert_eq!(
        block_on(provider.read(Path::new("a.fea"))).unwrap(),
        b"old-a"
    );
    assert!(!directory.exists());
    assert!(block_on(provider.lookup_workspace_publication(&requested))
        .unwrap()
        .is_none());
}

#[test]
fn interrupted_committed_publication_recovers_before_the_next_product_read() {
    let (root, provider) = setup();
    let _ = request(&provider); // Initializes the admitted mount journal.
    let directory;
    {
        let _guard = guard(root.path(), true).unwrap();
        directory = commit_without_materialization(root.path(), WorkspacePublicationId::new());
        // Inject process loss after one physical update. The durable commit and
        // pending record remain exactly as during a partially materialized batch.
        fs::write(root.path().join("a.fea"), b"new-a").unwrap();
    }
    drop(provider);
    let restored = SandboxFsProvider::new(root.path().to_path_buf()).unwrap();
    assert_eq!(
        block_on(restored.read(Path::new("b.fea"))).unwrap(),
        b"new-b"
    );
    assert_eq!(
        block_on(restored.read(Path::new("a.fea"))).unwrap(),
        b"new-a"
    );
    assert!(directory.join("complete").exists());
    assert!(!directory.join("plan.json").exists());
    assert!(!root.path().join(JOURNAL).join("pending.json").exists());
}

#[test]
fn corrupted_recovery_content_fails_closed_before_returning_a_source_read() {
    let (root, provider) = setup();
    let _ = request(&provider);
    {
        let _guard = guard(root.path(), true).unwrap();
        let directory = commit_without_materialization(root.path(), WorkspacePublicationId::new());
        fs::write(directory.join("plan.json"), b"[]").unwrap();
    }
    assert!(block_on(provider.read(Path::new("a.fea"))).is_err());
    assert_eq!(fs::read(root.path().join("a.fea")).unwrap(), b"old-a");
}

#[test]
fn uncommitted_preparation_is_discarded_without_changing_source() {
    let (root, provider) = setup();
    let _ = request(&provider);
    let id = WorkspacePublicationId::new();
    let directory = journal::record_directory(root.path(), id);
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("plan.json"), b"partial preparation").unwrap();
    crate::native::atomic_create_new_native(
        &root.path().join(JOURNAL).join("pending.json"),
        &serde_json::to_vec(&id).unwrap(),
    )
    .unwrap();
    assert_eq!(
        block_on(provider.read(Path::new("a.fea"))).unwrap(),
        b"old-a"
    );
    assert!(!directory.exists());
}

#[test]
fn old_writable_handle_cannot_acknowledge_writes_into_a_replaced_inode() {
    let (_root, provider) = setup();
    let input = request(&provider);
    let mut file = provider
        .open(
            Path::new("a.fea"),
            &OpenFlags {
                read: true,
                write: true,
                ..Default::default()
            },
        )
        .unwrap();
    block_on(provider.publish_workspace(input)).unwrap();
    assert_eq!(
        file.write_all(b"lost edit").unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        block_on(provider.read(Path::new("a.fea"))).unwrap(),
        b"new-a"
    );
}

#[test]
fn competing_native_publishers_have_one_winner_and_no_partial_capture() {
    use std::sync::{Arc, Barrier};
    let (root, provider) = setup();
    let mut input = request(&provider);
    input.edits = vec![
        WorkspaceEdit::Write {
            path: "a.fea".into(),
            bytes: b"new-a".to_vec(),
        },
        WorkspaceEdit::Write {
            path: "b.fea".into(),
            bytes: b"new-b".to_vec(),
        },
    ];
    let barrier = Arc::new(Barrier::new(9));
    std::thread::scope(|scope| {
        let mut publishers = Vec::new();
        for _ in 0..8 {
            let mut input = input.clone();
            input.id = WorkspacePublicationId::new();
            let barrier = barrier.clone();
            let root = root.path().to_path_buf();
            publishers.push(scope.spawn(move || {
                let provider = SandboxFsProvider::new(root).unwrap();
                barrier.wait();
                for _ in 0..200 {
                    match block_on(provider.publish_workspace(input.clone())) {
                        Ok(WorkspacePublicationOutcome::Published { .. }) => return 1,
                        Ok(WorkspacePublicationOutcome::Conflict { .. }) => return 0,
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(2))
                        }
                        other => panic!("unexpected publisher result: {other:?}"),
                    }
                }
                panic!("native publishers did not settle");
            }));
        }
        barrier.wait();
        for _ in 0..20 {
            match block_on(provider.capture_workspace(WorkspaceCaptureRequest {
                root: "/".into(),
                excluded_subtrees: vec![".artifacts".into()],
                max_files: 100,
                max_bytes: 1024,
            })) {
                Ok(snapshot) => {
                    let a = &snapshot
                        .files
                        .iter()
                        .find(|file| file.path == Path::new("a.fea"))
                        .unwrap()
                        .bytes;
                    let b = &snapshot
                        .files
                        .iter()
                        .find(|file| file.path == Path::new("b.fea"))
                        .unwrap()
                        .bytes;
                    assert!((a == b"old-a" && b == b"old-b") || (a == b"new-a" && b == b"new-b"));
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(2))
                }
                other => panic!("unexpected capture result: {other:?}"),
            }
        }
        assert_eq!(
            publishers
                .into_iter()
                .map(|publisher| publisher.join().unwrap())
                .sum::<usize>(),
            1
        );
    });
}

#[cfg(unix)]
#[test]
fn symlink_escape_is_rejected_for_capture_and_ordinary_product_writes() {
    let (root, provider) = setup();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
    assert!(
        block_on(provider.capture_workspace(WorkspaceCaptureRequest {
            root: "/".into(),
            excluded_subtrees: vec![".artifacts".into()],
            max_files: 100,
            max_bytes: 1024
        }))
        .is_err()
    );
    assert_eq!(
        block_on(provider.write(Path::new("escape/outside.fea"), b"bad"))
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(!outside.path().join("outside.fea").exists());
}
