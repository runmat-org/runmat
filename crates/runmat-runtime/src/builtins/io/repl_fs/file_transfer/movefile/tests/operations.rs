use super::*;
use runmat_builtins::MOVEFILE_RESULT_DEST_EXISTS;
use std::fs::{self, File};
use tempfile::tempdir;

#[test]
fn renames_file_and_removes_source_path() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source.txt");
    let target = temp.path().join("target.txt");
    File::create(&source).unwrap();
    let outcome = evaluate(&[text(&source), text(&target)]).unwrap();
    assert_eq!(outcome.status(), 1.0);
    assert!(!source.exists() && target.exists());
    let outputs = outcome.outputs();
    assert!(matches!(outputs[0], Value::Num(1.0)));
    assert!(matches!(outputs[1], Value::CharArray(ref value) if value.cols == 0));
    assert!(matches!(outputs[2], Value::CharArray(ref value) if value.cols == 0));
}

#[test]
fn moves_file_into_existing_directory() {
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
    assert!(!source.exists() && target.join("report.txt").exists());
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
        MOVEFILE_RESULT_DEST_EXISTS.identifier.unwrap()
    );
    assert!(source.exists() && target.exists());
    let replaced = evaluate(&[text(&source), text(&target), Value::from("f")]).unwrap();
    assert_eq!(replaced.status(), 1.0);
    assert!(!source.exists() && target.exists());
}

#[test]
fn wildcard_moves_every_match() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let target = temp.path().join("logs");
    fs::create_dir(&target).unwrap();
    let first = temp.path().join("a.log");
    let second = temp.path().join("b.log");
    File::create(&first).unwrap();
    File::create(&second).unwrap();
    assert_eq!(
        evaluate(&[text(&temp.path().join("*.log")), text(&target)])
            .unwrap()
            .status(),
        1.0
    );
    assert!(!first.exists() && !second.exists());
    assert!(target.join("a.log").exists() && target.join("b.log").exists());
}

#[test]
fn identical_path_and_containing_directory_are_successful_noops() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let directory = temp.path().join("docs");
    fs::create_dir(&directory).unwrap();
    let source = directory.join("readme.txt");
    File::create(&source).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&source)]).unwrap().status(),
        1.0
    );
    assert_eq!(
        evaluate(&[text(&source), text(&directory)])
            .unwrap()
            .status(),
        1.0
    );
    assert!(source.exists());
}
