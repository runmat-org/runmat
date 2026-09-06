use std::fs::File;

use runmat_value::Value;
use tempfile::tempdir;

use super::{evaluate, lock};

#[test]
fn rejects_non_text_and_empty_names_before_filesystem_access() {
    let non_text = evaluate(vec![Value::Num(1.0)]).expect_err("non-text path rejects");
    assert_eq!(non_text.identifier(), Some("RunMat:mkdir:InvalidFolder"));

    let empty = evaluate(vec![Value::from("")]).expect_err("empty path rejects");
    assert_eq!(empty.identifier(), Some("RunMat:mkdir:InvalidFolderName"));
}

#[test]
fn rejects_a_rooted_child_in_the_two_input_form() {
    let temp = tempdir().expect("temporary directory");
    let error = evaluate(vec![
        Value::from(temp.path().to_string_lossy().to_string()),
        Value::from(std::path::MAIN_SEPARATOR.to_string()),
    ])
    .expect_err("rooted child rejects");

    assert_eq!(
        error.identifier(),
        Some("RunMat:mkdir:FolderMustBeRelative")
    );
}

#[test]
fn preserves_a_file_at_the_target_and_returns_failure() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("occupied");
    File::create(&target).expect("seed file");

    let outcome = evaluate(vec![Value::from(target.to_string_lossy().to_string())])
        .expect("operational failures are outcomes");

    assert!(!outcome.status());
    assert_eq!(outcome.identifier(), "RunMat:mkdir:TargetNotDirectory");
    assert!(target.is_file());
}
