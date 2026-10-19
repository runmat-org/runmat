use super::execute;
use runmat_builtins::MATLABROOT_ERROR_TOO_MANY_INPUTS;
use runmat_value::{CharArray, Value};
use std::convert::TryFrom;

#[test]
fn returns_a_nonempty_character_row() {
    let value = execute::run(Vec::new()).expect("matlabroot");
    assert!(matches!(value, Value::CharArray(CharArray { rows: 1, .. })));
    assert!(!String::try_from(&value).expect("character row").is_empty());
}

#[test]
fn rejects_inputs_with_the_catalog_error() {
    let error = execute::run(vec![Value::Num(1.0)]).expect_err("arity error");
    assert_eq!(error.message(), MATLABROOT_ERROR_TOO_MANY_INPUTS.message);
    assert_eq!(
        error.identifier.as_deref(),
        MATLABROOT_ERROR_TOO_MANY_INPUTS.identifier
    );
}
