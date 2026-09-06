use runmat_value::{CharArray, StringArray, Tensor, Value};
use tempfile::tempdir;

#[test]
fn rejects_excess_arity_and_output_count() {
    let _lock = super::support::lock();
    let input_error = super::support::run(
        vec![Value::String("a".into()), Value::String("b".into())],
        1,
        true,
    )
    .expect_err("inputs");
    assert_eq!(
        input_error.identifier(),
        runmat_builtins::SAVEPATH_ERROR_TOO_MANY_INPUTS.identifier
    );

    let output_error = super::support::run(Vec::new(), 4, true).expect_err("outputs");
    assert_eq!(
        output_error.identifier(),
        runmat_builtins::SAVEPATH_ERROR_TOO_MANY_OUTPUTS.identifier
    );
}

#[test]
fn rejects_empty_and_non_scalar_text_before_writing() {
    let _lock = super::support::lock();
    let empty =
        super::support::run(vec![Value::String(String::new())], 1, true).expect_err("empty");
    assert_eq!(
        empty.identifier(),
        runmat_builtins::SAVEPATH_ERROR_EMPTY_FILENAME.identifier
    );

    let strings = StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).expect("strings");
    let string_error =
        super::support::run(vec![Value::StringArray(strings)], 1, true).expect_err("string array");
    assert_eq!(
        string_error.identifier(),
        runmat_builtins::SAVEPATH_ERROR_ARGUMENT_TYPE.identifier
    );

    let chars = CharArray::new("abcd".chars().collect(), 2, 2).expect("chars");
    let char_error =
        super::support::run(vec![Value::CharArray(chars)], 1, true).expect_err("char matrix");
    assert_eq!(
        char_error.identifier(),
        runmat_builtins::SAVEPATH_ERROR_ARGUMENT_TYPE.identifier
    );
}

#[test]
fn rejects_invalid_numeric_codes_and_reports_write_failure_as_status() {
    let _lock = super::support::lock();
    let fractional = Tensor::new(vec![65.5], vec![1, 1]).expect("tensor");
    let type_error =
        super::support::run(vec![Value::Tensor(fractional)], 1, true).expect_err("fractional");
    assert_eq!(
        type_error.identifier(),
        runmat_builtins::SAVEPATH_ERROR_ARGUMENT_TYPE.identifier
    );

    let directory = tempdir().expect("directory");
    let parent_file = directory.path().join("occupied");
    std::fs::write(&parent_file, "file").expect("parent file");
    let target = parent_file.join("pathdef.m");
    let value = super::support::run(
        vec![Value::String(target.to_string_lossy().into_owned())],
        1,
        false,
    )
    .expect("status result");
    assert_eq!(super::support::status(&value), 1.0);
}

#[test]
fn empty_default_override_is_a_status_failure() {
    let _lock = super::support::lock();
    let _environment = super::support::EnvironmentGuard::set_text("");
    let value = super::support::run(Vec::new(), 1, false).expect("status result");
    assert_eq!(super::support::status(&value), 1.0);
}
