use std::fs::File;

use runmat_value::{IntValue, Value};
use tempfile::tempdir;

use super::{evaluate, lock};

#[test]
fn validates_the_folder_and_option_grammar() {
    let folder = Value::from("folder");
    for args in [
        vec![],
        vec![Value::Num(1.0)],
        vec![folder.clone(), Value::from("recursive")],
        vec![folder.clone(), Value::from("ResolveSymbolicLinks")],
        vec![
            folder.clone(),
            Value::from("ResolveSymbolicLinks"),
            Value::Num(2.0),
        ],
    ] {
        assert!(evaluate(args).is_err());
    }
}

#[test]
fn accepts_exact_integer_symbolic_link_controls() {
    let temp = tempdir().expect("temporary directory");
    let missing = temp.path().join("missing");
    let outcome = evaluate(vec![
        Value::from(missing.to_string_lossy().to_string()),
        Value::from("ResolveSymbolicLinks"),
        Value::Int(IntValue::U64(0)),
    ])
    .expect("exact integer option parses");
    assert_eq!(outcome.identifier(), "RunMat:rmdir:DirectoryNotFound");
}

#[test]
fn rejects_files_without_removing_them() {
    let _lock = lock();
    let temp = tempdir().expect("temporary directory");
    let target = temp.path().join("file.txt");
    File::create(&target).expect("seed file");

    let outcome = evaluate(vec![Value::from(target.to_string_lossy().to_string())])
        .expect("operational failures are outcomes");
    assert_eq!(outcome.identifier(), "RunMat:rmdir:NotADirectory");
    assert!(target.is_file());
}
