//! Real directory permissions obstruct synchronization after atomic visibility.
use super::*;
use std::os::unix::fs::PermissionsExt;

struct Restore(PathBuf, fs::Permissions);
impl Drop for Restore {
    fn drop(&mut self) {
        fs::set_permissions(&self.0, self.1.clone()).unwrap();
    }
}

#[test]
fn visible_unsynchronized_commit_blocks_reads_and_preserves_exact_retry_receipt() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("original"), b"old").unwrap();
    let request = WorkspacePublicationRequest {
        id: WorkspacePublicationId::new(),
        root: "/".into(),
        excluded_subtrees: vec![".artifacts".into()],
        expected_revision: WorkspaceRevision([3; 32]),
        edits: vec![WorkspaceEdit::Write {
            path: "original".into(),
            bytes: b"new".to_vec(),
        }],
    };
    let directory = record_directory(root.path(), request.id);
    let mut record = Record {
        digest: publication_digest(&request).unwrap(),
        plan_digest: [0; 32],
        receipt: WorkspacePublicationReceipt {
            id: request.id,
            revision: WorkspaceRevision([7; 32]),
        },
    };
    let fence = guard(root.path(), true).unwrap();
    let mut restore = None;
    let durability = write_record_checked(
        &directory,
        &[FinalEdit::Write {
            path: "original".into(),
            bytes: b"new".to_vec(),
            mode: None,
        }],
        &mut record,
        || {
            restore = Some(Restore(
                directory.clone(),
                fs::metadata(&directory)?.permissions(),
            ));
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o300))
        },
    )
    .unwrap();
    assert_eq!(durability, CommitDurability::Indeterminate);
    drop(fence);
    let error = guard(root.path(), false)
        .err()
        .expect("unconfirmed durability exposed source");
    assert_eq!(uncertain_receipt(&error), Some(&record.receipt));
    assert_eq!(fs::read(root.path().join("original")).unwrap(), b"old");
    let expected = WorkspacePublicationOutcome::DurabilityIndeterminate {
        receipt: record.receipt.clone(),
    };
    assert_eq!(
        crate::workspace_native::lookup(root.path(), &request).unwrap(),
        Some(expected.clone())
    );
    assert_eq!(
        futures::executor::block_on(crate::workspace_native::publish(
            root.path(),
            request.clone()
        ))
        .unwrap(),
        expected
    );
    let mut substituted = request.clone();
    substituted.edits.clear();
    assert_eq!(
        crate::workspace_native::lookup(root.path(), &substituted).unwrap(),
        Some(WorkspacePublicationOutcome::IdReused)
    );
    substituted.id = WorkspacePublicationId::new();
    assert!(crate::workspace_native::lookup(root.path(), &substituted).is_err());
    assert!(!directory.join("complete").exists());
    drop(restore);
    assert_eq!(
        crate::workspace_native::lookup(root.path(), &request).unwrap(),
        Some(WorkspacePublicationOutcome::Published {
            receipt: record.receipt
        })
    );
    assert_eq!(fs::read(root.path().join("original")).unwrap(), b"new");
    assert!(directory.join("complete").exists());
}
