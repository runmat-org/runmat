use runmat_value::{CharArray, Value};
use tempfile::{tempdir, NamedTempFile};

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[test]
fn rejects_missing_arguments_and_conflicting_positions() {
    let _guard = PathGuard::new();
    assert_eq!(
        super::run(Vec::new()).expect_err("missing").identifier(),
        runmat_builtins::ADDPATH_ERROR_TOO_FEW_ARGUMENTS.identifier
    );
    let directory = tempdir().expect("directory");
    let error = super::run(vec![
        Value::String(directory.path().to_string_lossy().into_owned()),
        Value::String("-begin".into()),
        Value::String("-end".into()),
    ])
    .expect_err("conflicting positions");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ADDPATH_ERROR_POSITION.identifier
    );
}

#[test]
fn failed_validation_leaves_the_path_unchanged() {
    let _guard = PathGuard::new();
    let valid = tempdir().expect("valid");
    let original = canonical(valid.path());
    crate::builtins::common::path_state::set_path_string(&original);
    let error = super::run(vec![
        Value::String(valid.path().to_string_lossy().into_owned()),
        Value::String("missing/path/for/addpath".into()),
    ])
    .expect_err("missing folder");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ADDPATH_ERROR_FOLDER_NOT_FOUND.identifier
    );
    assert_eq!(
        crate::builtins::common::path_state::current_path_string(),
        original
    );
}

#[test]
fn rejects_character_arrays_with_more_than_two_dimensions() {
    let _guard = PathGuard::new();
    let original = crate::builtins::common::path_state::current_path_string();
    let chars = CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).expect("3-D chars");
    let error = super::run(vec![Value::CharArray(chars)]).expect_err("invalid characters");
    assert_eq!(
        error.identifier(),
        runmat_builtins::ADDPATH_ERROR_ARGUMENT_TYPE.identifier
    );
    assert_eq!(
        crate::builtins::common::path_state::current_path_string(),
        original
    );
}

#[test]
fn distinguishes_files_and_the_reserved_path_definition_name() {
    let _guard = PathGuard::new();
    let file = NamedTempFile::new().expect("file");
    let file_error = super::run(vec![Value::String(
        file.path().to_string_lossy().into_owned(),
    )])
    .expect_err("not a folder");
    assert_eq!(
        file_error.identifier(),
        runmat_builtins::ADDPATH_ERROR_NOT_FOLDER.identifier
    );

    let pathdef_error =
        super::run(vec![Value::String("pathdef".into())]).expect_err("reserved path name");
    assert_eq!(
        pathdef_error.identifier(),
        runmat_builtins::ADDPATH_ERROR_PATHDEF.identifier
    );
}
