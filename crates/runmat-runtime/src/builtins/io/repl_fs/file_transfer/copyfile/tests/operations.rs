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
