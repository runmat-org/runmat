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

#[test]
fn force_preserves_self_through_canonical_path_aliases() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("stable.txt");
    fs::write(&source, b"preserved").unwrap();
    let alias = temp.path().join(".").join("stable.txt");
    assert_eq!(
        evaluate(&[text(&source), text(&alias), Value::from("f")])
            .unwrap()
            .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
    // Wildcard planning must protect aliases before any replacement is attempted.
    assert_eq!(
        evaluate(&[
            text(&temp.path().join("*.txt")),
            text(&temp.path().join(".")),
            Value::from("f")
        ])
        .unwrap()
        .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
}

#[cfg(unix)]
#[test]
fn force_preserves_symbolic_alias_and_moves_dangling_source_link() {
    use std::os::unix::fs::symlink;
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source");
    let alias = temp.path().join("alias");
    fs::write(&source, b"preserved").unwrap();
    symlink(&source, &alias).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&alias), Value::from("f")])
            .unwrap()
            .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
    assert!(fs::symlink_metadata(&alias)
        .unwrap()
        .file_type()
        .is_symlink());
    let dangling = temp.path().join("dangling");
    let moved = temp.path().join("moved");
    let absent = temp.path().join("absent");
    symlink(&absent, &dangling).unwrap();
    assert_eq!(
        evaluate(&[text(&dangling), text(&moved)]).unwrap().status(),
        1.0
    );
    assert!(fs::symlink_metadata(&dangling).is_err());
    assert_eq!(fs::read_link(&moved).unwrap(), absent);
}

#[test]
fn rejects_descendant_target_before_force_removal() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("tree");
    let destination = source.join("nested");
    let collision = destination.join("tree");
    fs::create_dir_all(&collision).unwrap();
    fs::write(collision.join("preserve"), b"payload").unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&destination), Value::from("f")])
            .unwrap()
            .status(),
        0.0
    );
    assert_eq!(fs::read(collision.join("preserve")).unwrap(), b"payload");
}
