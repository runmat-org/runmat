use super::*;
use runmat_builtins::{
    COPYFILE_ERROR_FLAG_ARG, COPYFILE_RESULT_EMPTY_DEST, COPYFILE_RESULT_EMPTY_SOURCE,
    COPYFILE_RESULT_PATTERN_ERROR, COPYFILE_RESULT_SAME_PATH, COPYFILE_RESULT_SOURCE_NOT_FOUND,
};
use runmat_value::CharArray;
use std::fs::File;
use tempfile::tempdir;

#[test]
fn reports_missing_empty_and_same_path_inputs() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let missing = temp.path().join("missing.txt");
    let target = temp.path().join("target.txt");
    let outcome = evaluate(&[text(&missing), text(&target)]).unwrap();
    assert_eq!(
        outcome.message_id(),
        COPYFILE_RESULT_SOURCE_NOT_FOUND.identifier.unwrap()
    );
    assert!(outcome.message().contains("does not exist"));
    let empty_source = evaluate(&[Value::from(""), text(&target)]).unwrap();
    assert_eq!(
        empty_source.message_id(),
        COPYFILE_RESULT_EMPTY_SOURCE.identifier.unwrap()
    );
    File::create(&target).unwrap();
    let empty_target = evaluate(&[text(&target), Value::from("")]).unwrap();
    assert_eq!(
        empty_target.message_id(),
        COPYFILE_RESULT_EMPTY_DEST.identifier.unwrap()
    );
    let same = evaluate(&[text(&target), text(&target)]).unwrap();
    assert_eq!(
        same.message_id(),
        COPYFILE_RESULT_SAME_PATH.identifier.unwrap()
    );
}

#[test]
fn validates_force_flag_and_accepts_uppercase_character_row() {
    let error = evaluate(&[Value::from("a"), Value::from("b"), Value::Num(1.0)]).unwrap_err();
    assert_eq!(error.message(), COPYFILE_ERROR_FLAG_ARG.message);
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source.txt");
    let target = temp.path().join("target.txt");
    File::create(&source).unwrap();
    File::create(&target).unwrap();
    let outcome = evaluate(&[
        text(&source),
        text(&target),
        Value::CharArray(CharArray::new_row("F")),
    ])
    .unwrap();
    assert_eq!(outcome.status(), 1.0);
}

#[test]
fn rejects_invalid_wildcard_without_mutation() {
    let outcome = evaluate(&[Value::from("[*.txt"), Value::from("dest")]).unwrap();
    assert_eq!(outcome.status(), 0.0);
    assert_eq!(
        outcome.message_id(),
        COPYFILE_RESULT_PATTERN_ERROR.identifier.unwrap()
    );
}
