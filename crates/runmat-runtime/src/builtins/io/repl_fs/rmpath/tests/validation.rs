use runmat_value::{IntValue, Tensor, Value};
use tempfile::{tempdir, NamedTempFile};

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn rejects_every_numeric_form_before_provider_access() {
    let _guard = PathGuard::new();
    let inputs = [
        Value::Int(IntValue::U8(1)),
        Value::Tensor(Tensor::new(vec![65.0], vec![1, 1]).expect("tensor")),
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: 0,
            buffer_id: 9_446_001,
            descriptor: Default::default(),
        }),
    ];
    for input in inputs {
        let error = super::run(vec![input]).expect_err("numeric rejection");
        assert_eq!(
            error.identifier(),
            runmat_builtins::RMPATH_ERROR_ARGUMENT_TYPE.identifier
        );
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }
}

#[test]
fn failed_removal_leaves_the_path_unchanged() {
    let _guard = PathGuard::new();
    let first = tempdir().expect("first");
    let second = tempdir().expect("second");
    let first = canonical(first.path());
    let second = canonical(second.path());
    let original = format!(
        "{first}{}{second}",
        crate::builtins::common::path_state::PATH_LIST_SEPARATOR
    );
    crate::builtins::common::path_state::set_path_string(&original);
    let error = super::run(vec![
        Value::String(first),
        Value::String("missing/path/for/rmpath".into()),
    ])
    .expect_err("missing folder");
    assert_eq!(
        error.identifier(),
        runmat_builtins::RMPATH_ERROR_FOLDER_NOT_FOUND.identifier
    );
    assert_eq!(
        crate::builtins::common::path_state::current_path_string(),
        original
    );
}

#[test]
fn distinguishes_existing_folders_that_are_not_on_the_path() {
    let _guard = PathGuard::new();
    let directory = tempdir().expect("directory");
    crate::builtins::common::path_state::set_path_string("");
    let error = super::run(vec![Value::String(
        directory.path().to_string_lossy().into_owned(),
    )])
    .expect_err("not on path");
    assert_eq!(
        error.identifier(),
        runmat_builtins::RMPATH_ERROR_NOT_ON_PATH.identifier
    );
}

#[test]
fn distinguishes_files_from_missing_folders() {
    let _guard = PathGuard::new();
    let file = NamedTempFile::new().expect("file");
    crate::builtins::common::path_state::set_path_string("");
    let error = super::run(vec![Value::String(
        file.path().to_string_lossy().into_owned(),
    )])
    .expect_err("not a folder");
    assert_eq!(
        error.identifier(),
        runmat_builtins::RMPATH_ERROR_NOT_FOLDER.identifier
    );
}
