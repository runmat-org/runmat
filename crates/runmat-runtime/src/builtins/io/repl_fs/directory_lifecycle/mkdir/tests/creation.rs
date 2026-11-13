use std::fs;

use runmat_value::Value;
use tempfile::tempdir;

use super::{evaluate, invoke, lock};

#[test]
fn creates_a_directory_and_returns_logical_status() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("single");

    let status =
        invoke(vec![Value::from(target.to_string_lossy().to_string())]).expect("mkdir succeeds");

    assert_eq!(status, Value::Bool(true));
    assert!(target.is_dir());
}

#[test]
fn creates_missing_parent_and_nested_child() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let parent = temp.path().join("missing");

    let outcome = evaluate(vec![
        Value::from(parent.to_string_lossy().to_string()),
        Value::from("archive/2026"),
    ])
    .expect("mkdir evaluates");

    assert!(outcome.status());
    assert!(parent.join("archive/2026").is_dir());
}

#[test]
fn reports_an_existing_directory_as_a_notice() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("existing");
    fs::create_dir(&target).expect("seed directory");

    let outcome =
        evaluate(vec![Value::from(target.to_string_lossy().to_string())]).expect("mkdir evaluates");

    assert!(outcome.status());
    assert_eq!(outcome.message(), "Directory already exists.");
    assert_eq!(outcome.identifier(), "RunMat:mkdir:DirectoryExists");
}
