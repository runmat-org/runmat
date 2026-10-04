use super::*;
use crate::{FsProvider, NativeFsProvider};

fn pending(root: &Path, plan: &[FinalEdit]) -> (PathBuf, WorkspacePublicationReceipt) {
    let _guard = guard(root, true).unwrap();
    let id = WorkspacePublicationId::new();
    let directory = record_directory(root, id);
    let mut record = Record {
        digest: [1; 32],
        plan_digest: [0; 32],
        receipt: WorkspacePublicationReceipt {
            id,
            revision: WorkspaceRevision([2; 32]),
        },
    };
    write_record(&directory, plan, &mut record).unwrap();
    (directory, record.receipt)
}

#[test]
fn cold_scope_rejects_dense_recovery_before_mutation_then_recovers_with_original_receipt() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("original"), b"old").unwrap();
    let original = vec![42; 32 * 1024];
    let (directory, receipt) = pending(
        root.path(),
        &[FinalEdit::Write {
            path: "original".into(),
            bytes: original.clone(),
            mode: None,
        }],
    );
    assert!(NativeFsProvider
        .workspace_access(root.path(), false, 1 << 20)
        .is_err());
    assert_eq!(fs::read(root.path().join("original")).unwrap(), b"old");
    assert!(directory.join("plan.json").exists());
    assert!(!directory.join("complete").exists());
    assert_eq!(read_record(&directory).unwrap().unwrap().receipt, receipt);
    let scope = NativeFsProvider
        .workspace_access(root.path(), false, 128 << 20)
        .unwrap();
    assert_eq!(
        scope.read(Path::new("original"), 1 << 20).unwrap(),
        original
    );
    assert_eq!(read_record(&directory).unwrap().unwrap().receipt, receipt);
    assert!(directory.join("complete").exists());
    assert!(!directory.join("plan.json").exists());
}

#[test]
fn sparse_oversized_recovery_rejects_wire_allocation_before_digest_and_source_mutation() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("original"), b"old").unwrap();
    let (directory, _) = pending(
        root.path(),
        &[FinalEdit::Write {
            path: "original".into(),
            bytes: b"new".to_vec(),
            mode: None,
        }],
    );
    fs::OpenOptions::new()
        .write(true)
        .open(directory.join("plan.json"))
        .unwrap()
        .set_len(1 << 30)
        .unwrap();
    let error = match NativeFsProvider.workspace_access(root.path(), false, 1 << 20) {
        Ok(_) => panic!("oversized recovery admitted"),
        Err(error) => error,
    };
    let limit = crate::workspace_access::resource_limit(&error)
        .expect("recovery must retain its structured wire limit");
    assert_eq!(limit.required_bytes, 1 << 30);
    assert!(limit.maximum_bytes <= 1 << 20);
    assert_eq!(fs::read(root.path().join("original")).unwrap(), b"old");
    assert!(!directory.join("complete").exists());
}

#[test]
fn late_invalid_recovery_path_rejects_before_the_first_valid_edit() {
    let root = tempfile::tempdir().unwrap();
    fs::write(root.path().join("original"), b"old").unwrap();
    let (directory, _) = pending(
        root.path(),
        &[
            FinalEdit::Write {
                path: "original".into(),
                bytes: b"new".to_vec(),
                mode: None,
            },
            FinalEdit::Remove {
                path: "../escape".into(),
            },
        ],
    );
    assert!(NativeFsProvider
        .workspace_access(root.path(), false, 1 << 20)
        .is_err());
    assert_eq!(fs::read(root.path().join("original")).unwrap(), b"old");
    assert!(directory.join("plan.json").exists());
    assert!(!directory.join("complete").exists());
}
