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

#[test]
fn text_admission_precedes_resident_control_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let error = evaluate(vec![
        Value::from("unused"),
        Value::from("invalid-option"),
        resident,
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some(runmat_builtins::RMDIR_ERROR_OPTION.identifier.unwrap())
    );
    let pages =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap();
    let error = evaluate(vec![Value::CharArray(pages)]).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::RMDIR_ERROR_FOLDER_TYPE.identifier
    );
}

#[test]
fn no_output_removal_executes_and_reports_missing_target() {
    let _lock = lock();
    let temp = tempdir().unwrap();
    let path = temp.path().join("removed");
    std::fs::create_dir(&path).unwrap();
    let args = vec![Value::from(path.to_string_lossy().to_string())];
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    assert_eq!(
        super::invoke(args.clone()).unwrap(),
        Value::output_list(vec![])
    );
    assert!(!path.exists());
    assert!(super::invoke(args).is_err());
}
