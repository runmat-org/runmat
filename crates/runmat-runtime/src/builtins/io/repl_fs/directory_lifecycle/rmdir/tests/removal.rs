use std::fs::{self, File};

use runmat_value::Value;
use tempfile::tempdir;

use super::{evaluate, invoke, lock};

#[test]
fn removes_an_empty_directory_and_returns_logical_status() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("empty");
    fs::create_dir(&target).expect("seed directory");

    let status =
        invoke(vec![Value::from(target.to_string_lossy().to_string())]).expect("rmdir succeeds");

    assert_eq!(status, Value::Bool(true));
    assert!(!target.exists());
}

#[test]
fn recursive_mode_removes_a_hierarchy() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("tree");
    fs::create_dir_all(target.join("nested")).expect("seed hierarchy");
    File::create(target.join("nested/data.bin")).expect("seed file");

    let outcome = evaluate(vec![
        Value::from(target.to_string_lossy().to_string()),
        Value::from("S"),
    ])
    .expect("rmdir evaluates");

    assert!(outcome.status());
    assert!(!target.exists());
}

#[test]
fn nonrecursive_mode_preserves_a_nonempty_directory() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("nonempty");
    fs::create_dir(&target).expect("seed directory");
    File::create(target.join("data.bin")).expect("seed file");

    let outcome = evaluate(vec![Value::from(target.to_string_lossy().to_string())])
        .expect("operational failures are outcomes");

    assert!(!outcome.status());
    assert_eq!(outcome.identifier(), "RunMat:rmdir:DirectoryNotEmpty");
    assert!(target.is_dir());
}
