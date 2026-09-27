use super::idivide_builtin;
use futures::executor::block_on;
use runmat_value::{IntValue, Value};

#[test]
fn rounding_modes_preserve_the_integer_class() {
    for (mode, expected) in [("fix", -2), ("floor", -3), ("ceil", -2), ("round", -2)] {
        let result = block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(-7)),
            Value::Int(IntValue::I16(3)),
            Value::String(mode.into()),
        ]))
        .expect("idivide");
        assert_eq!(result, Value::Int(IntValue::I16(expected)));
    }
}

#[test]
fn extreme_scalar_double_overflow_is_structured_in_every_rounding_mode() {
    for mode in ["fix", "floor", "ceil", "round"] {
        let error = block_on(idivide_builtin(vec![
            Value::Num(-1e100),
            Value::Int(IntValue::I16(-1)),
            Value::from(mode),
        ]))
        .expect_err("extreme quotient must not panic or wrap");
        assert_eq!(
            error.identifier(),
            runmat_builtins::IDIVIDE_ERROR_OVERFLOW.identifier
        );
    }
}

#[test]
fn requested_output_admission_preserves_discard_and_single_output() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count);
        assert_eq!(
            block_on(idivide_builtin(vec![
                Value::Int(IntValue::I16(6)),
                Value::Int(IntValue::I16(2)),
            ]))
            .expect("valid output count"),
            Value::Int(IntValue::I16(3)),
        );
    }
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(idivide_builtin(vec![
        Value::Int(IntValue::I16(6)),
        Value::Int(IntValue::I16(2)),
    ]))
    .expect_err("output admission precedes operand consumption");
    assert_eq!(
        error.identifier(),
        runmat_builtins::IDIVIDE_ERROR_INVALID_INPUT.identifier
    );
}

#[test]
fn rounding_character_rows_allow_trailing_singletons_not_pages() {
    for (shape, valid) in [(vec![1, 3, 1], true), (vec![1, 1, 3], false)] {
        let control =
            runmat_value::CharArray::new_with_shape("fix".chars().collect(), shape).unwrap();
        let result = block_on(idivide_builtin(vec![
            Value::Int(IntValue::I16(6)),
            Value::Int(IntValue::I16(2)),
            Value::CharArray(control),
        ]));
        assert_eq!(result.is_ok(), valid);
    }
}
