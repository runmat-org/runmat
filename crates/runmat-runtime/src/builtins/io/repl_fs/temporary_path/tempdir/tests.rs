use super::execute;
use runmat_builtins::TEMPDIR_ERROR_TOO_MANY_INPUTS;
use runmat_value::{CharArray, Value};
use std::convert::TryFrom;

#[test]
fn returns_a_stable_nonempty_character_row_with_separator() {
    let first = execute::run(Vec::new()).expect("tempdir");
    let second = execute::run(Vec::new()).expect("tempdir");
    assert_eq!(first, second);
    match &first {
        Value::CharArray(CharArray { rows: 1, cols, .. }) => assert!(*cols > 0),
        other => panic!("expected character row, got {other:?}"),
    }
    let text = String::try_from(&first).expect("character row");
    assert!(text.ends_with(std::path::MAIN_SEPARATOR));
}

#[test]
fn rejects_inputs_with_the_catalog_error() {
    let error = execute::run(vec![Value::Num(1.0)]).expect_err("arity error");
    assert_eq!(error.message(), TEMPDIR_ERROR_TOO_MANY_INPUTS.message);
}
