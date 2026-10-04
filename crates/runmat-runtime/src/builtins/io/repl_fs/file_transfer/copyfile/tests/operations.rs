use super::*;
use runmat_builtins::{COPYFILE_RESULT_DEST_EXISTS, COPYFILE_RESULT_DEST_MISSING};
use std::fs::{self, File};
use tempfile::tempdir;

#[test]
fn copies_file_to_new_name_and_retains_source() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source.txt");
    let target = temp.path().join("target.txt");
    File::create(&source).unwrap();
    let outcome = evaluate(&[text(&source), text(&target)]).unwrap();
    assert_eq!(outcome.status(), 1.0);
    assert!(source.exists() && target.exists());
    let outputs = outcome.outputs();
    assert!(matches!(outputs[0], Value::Num(1.0)));
    assert!(matches!(outputs[1], Value::CharArray(ref value) if value.cols == 0));
    assert!(matches!(outputs[2], Value::CharArray(ref value) if value.cols == 0));
}

#[test]
fn copies_file_into_existing_directory() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("report.txt");
    let target = temp.path().join("reports");
    File::create(&source).unwrap();
    fs::create_dir(&target).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&target)]).unwrap().status(),
        1.0
    );
    assert!(target.join("report.txt").exists());
}

#[test]
fn force_replaces_existing_file_and_default_preserves_it() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("draft.txt");
    let target = temp.path().join("final.txt");
    File::create(&source).unwrap();
    File::create(&target).unwrap();
    let preserved = evaluate(&[text(&source), text(&target)]).unwrap();
    assert_eq!(preserved.status(), 0.0);
    assert_eq!(
        preserved.message_id(),
        COPYFILE_RESULT_DEST_EXISTS.identifier.unwrap()
    );
    assert!(source.exists() && target.exists());
    let replaced = evaluate(&[text(&source), text(&target), Value::from("F")]).unwrap();
    assert_eq!(replaced.status(), 1.0);
    assert!(source.exists() && target.exists());
}

#[test]
fn copies_directory_tree() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("data");
    let target = temp.path().join("data-copy");
    fs::create_dir_all(source.join("raw")).unwrap();
    File::create(source.join("raw/sample.txt")).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&target)]).unwrap().status(),
        1.0
    );
    assert!(source.join("raw/sample.txt").exists());
    assert!(target.join("raw/sample.txt").exists());
}

#[test]
fn wildcard_copies_all_matches_to_existing_directory() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let target = temp.path().join("logs");
    fs::create_dir(&target).unwrap();
    File::create(temp.path().join("a.log")).unwrap();
    File::create(temp.path().join("b.log")).unwrap();
    let pattern = temp.path().join("*.log");
    assert_eq!(
        evaluate(&[text(&pattern), text(&target)]).unwrap().status(),
        1.0
    );
    assert!(target.join("a.log").exists() && target.join("b.log").exists());
}

#[test]
fn wildcard_requires_existing_destination_directory() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    File::create(temp.path().join("file.log")).unwrap();
    let outcome = evaluate(&[
        text(&temp.path().join("*.log")),
        text(&temp.path().join("missing")),
    ])
    .unwrap();
    assert_eq!(outcome.status(), 0.0);
    assert_eq!(
        outcome.message_id(),
        COPYFILE_RESULT_DEST_MISSING.identifier.unwrap()
    );
}

#[test]
fn captured_copy_owner_survives_global_provider_swap() {
    use runmat_filesystem::MemoryFsProvider;
    use std::sync::Arc;

    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/source.bin", b"selected")
        .expect("virtual source");
    let other = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    other
        .write_project_path("/selected/source.bin", b"wrong")
        .expect("lexically identical other-provider source");
    let captured = runmat_filesystem::with_provider_override(selected.clone(), || {
        super::super::CopyFilesystem::capture(None).expect("capture selected owner")
    });
    let _replacement = runmat_filesystem::replace_provider(other.clone());
    let outcome = futures::executor::block_on(super::super::single::copy(
        &captured,
        "source.bin",
        "target.bin",
        false,
    ));
    assert_eq!(outcome.status(), 1.0);
    assert_eq!(
        selected.read_project_path("/selected/target.bin").unwrap(),
        b"selected"
    );
    assert!(other.read_project_path("/other/target.bin").is_err());
}

#[test]
fn captured_copy_owner_applies_to_wildcards_and_recursive_children() {
    use runmat_filesystem::MemoryFsProvider;
    use std::sync::Arc;

    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
    selected
        .write_project_path("/selected/tree/child/item.bin", b"selected child")
        .expect("selected tree");
    selected
        .write_project_path("/selected/match-a.bin", b"selected wildcard")
        .expect("selected match");
    selected
        .write_project_path("/selected/matches/.keep", b"")
        .expect("selected destination");
    let other = Arc::new(MemoryFsProvider::with_current_dir("/other"));
    other
        .write_project_path("/selected/tree/child/item.bin", b"wrong child")
        .expect("other tree");
    other
        .write_project_path("/selected/match-a.bin", b"wrong wildcard")
        .expect("other match");
    let captured = runmat_filesystem::with_provider_override(selected.clone(), || {
        super::super::CopyFilesystem::capture(None).expect("capture selected owner")
    });
    let _replacement = runmat_filesystem::replace_provider(other.clone());
    let recursive = futures::executor::block_on(super::super::single::copy(
        &captured,
        "tree",
        "tree-copy",
        false,
    ));
    assert_eq!(recursive.status(), 1.0, "{recursive:?}");
    let wildcard = futures::executor::block_on(super::super::wildcard::copy(
        &captured,
        "match-*.bin",
        "matches",
        false,
    ));
    assert_eq!(wildcard.status(), 1.0, "{wildcard:?}");
    assert_eq!(
        selected
            .read_project_path("/selected/tree-copy/child/item.bin")
            .unwrap(),
        b"selected child"
    );
    assert_eq!(
        selected
            .read_project_path("/selected/matches/match-a.bin")
            .unwrap(),
        b"selected wildcard"
    );
    assert!(other
        .read_project_path("/selected/tree-copy/child/item.bin")
        .is_err());
    assert!(other
        .read_project_path("/selected/matches/match-a.bin")
        .is_err());
}

#[test]
fn cancelled_copy_rejects_before_provider_effects_with_typed_cause() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_filesystem::MemoryFsProvider;
    use std::rc::Rc;
    use std::sync::Arc;

    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let filesystem = Arc::new(MemoryFsProvider::with_current_dir("/workspace"));
    filesystem
        .write_project_path("/workspace/source.bin", b"unchanged")
        .expect("source");
    let cancellation = CancellationSource::new();
    cancellation.cancel(CancellationReason::User);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());
    let error = runmat_filesystem::with_provider_override(filesystem.clone(), || {
        futures::executor::block_on(super::super::copyfile_builtin(
            Some(runtime),
            vec![Value::from("source.bin"), Value::from("target.bin")],
        ))
        .expect_err("cancelled copy")
    });
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(error.source.as_deref().is_some_and(|cause| {
        matches!(
            cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        )
    }));
    assert_eq!(
        filesystem
            .read_project_path("/workspace/source.bin")
            .unwrap(),
        b"unchanged"
    );
    assert!(filesystem
        .read_project_path("/workspace/target.bin")
        .is_err());
}
