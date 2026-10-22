use runmat_value::{CharArray, Value};
use tempfile::{tempdir, NamedTempFile};

use super::super::super::path_mutation::test_support::{canonical, PathGuard};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn numeric_path_mutation_decode_keeps_low_rank_and_empty_row_policy() {
    use super::super::super::path_mutation::text::{self, DecodeError, NumericPolicy};

    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let policy =
        NumericPolicy::RunMatExtension(&runmat_builtins::ADDPATH_NUMERIC_CHARACTER_CODES_EXTENSION);
    for (shape, codes, expected) in [
        (Vec::new(), vec![97.0], "a"),
        (vec![1], vec![97.0], "a"),
        (vec![2], vec![97.0, 98.0], "ab"),
        (vec![0, 0], Vec::new(), ""),
    ] {
        let value = Value::Tensor(runmat_value::Tensor::new(codes, shape).unwrap());
        let Ok(tokens) = futures::executor::block_on(text::decode(
            crate::builtins::common::test_support::builtin_arguments(vec![value]),
            "addpath",
            policy,
        )) else {
            panic!("admitted numeric path row")
        };
        assert_eq!(tokens, vec![expected]);
    }
    for (shape, codes) in [
        (vec![2, 1], vec![97.0, 98.0]),
        (vec![1, 1, 2], vec![97.0, 98.0]),
    ] {
        let value = Value::Tensor(runmat_value::Tensor::new(codes, shape).unwrap());
        assert!(matches!(
            futures::executor::block_on(text::decode(
                crate::builtins::common::test_support::builtin_arguments(vec![value]),
                "addpath",
                policy
            )),
            Err(DecodeError::InvalidInput)
        ));
    }
    let allocation =
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![2]);
    assert!(matches!(
        futures::executor::block_on(text::decode(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                allocation.handle
            )]),
            "addpath",
            policy,
        )),
        Err(DecodeError::Provider(_))
    ));
}

#[test]
fn rejects_missing_arguments_and_conflicting_positions() {
    let _guard = PathGuard::new();
    assert_eq!(
        super::run(Vec::new()).expect_err("missing").identifier(),
        runmat_builtins::ADDPATH_ERROR_TOO_FEW_ARGUMENTS.identifier
    );
    let directory = tempdir().expect("directory");
    let error = super::run(vec![
        Value::String(directory.path().to_string_lossy().into_owned().into()),
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
    crate::builtins::common::path_state::set_path_string(&original)
        .expect("test search path update");
    let error = super::run(vec![
        Value::String(valid.path().to_string_lossy().into_owned().into()),
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
    let chars = CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
        .expect("3-D chars");
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
        file.path().to_string_lossy().into_owned().into(),
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
#[test]
fn invalid_later_numeric_codes_reject_before_first_resident_read() {
    let _guard = PathGuard::new();
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let invalid = Value::Tensor(runmat_value::Tensor::new(vec![65.5], vec![1, 1]).unwrap());
    let error = super::run(vec![resident, invalid]).unwrap_err();
    assert_eq!(
        error.identifier(),
        runmat_builtins::ADDPATH_ERROR_ARGUMENT_TYPE.identifier
    );
}
