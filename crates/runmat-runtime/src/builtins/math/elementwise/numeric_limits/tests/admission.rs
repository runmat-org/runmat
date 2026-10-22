use super::*;

#[test]
fn fixed_outputs_preserve_statement_and_scalar_result_contract() {
    for builtin in [flintmax, intmax, intmin, realmax, realmin] {
        for count in [None, Some(0), Some(1)] {
            let _outputs =
                crate::output_count::push_output_count(count).expect("output-count scope");
            assert!(builtin(vec![]).is_ok());
        }
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = builtin(vec![]).unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::NUMERIC_LIMIT_ERROR_INVALID_SYNTAX.identifier
        );
    }
}

#[test]
fn class_text_accepts_trailing_singletons_but_not_pages() {
    for (builtin, class) in [
        (flintmax as fn(Vec<Value>) -> BuiltinResult<Value>, "single"),
        (realmax, "single"),
        (realmin, "single"),
        (intmax, "uint64"),
        (intmin, "int64"),
    ] {
        let row = runmat_value::CharArray::from_unicode_scalars_with_shape(
            class.chars().collect(),
            vec![1, class.len(), 1],
        )
        .unwrap();
        assert!(builtin(vec![Value::CharArray(row)]).is_ok());
        let pages = runmat_value::CharArray::from_unicode_scalars_with_shape(
            class.chars().collect(),
            vec![1, 1, class.len()],
        )
        .unwrap();
        assert!(builtin(vec![Value::CharArray(pages)]).is_err());
    }
}
