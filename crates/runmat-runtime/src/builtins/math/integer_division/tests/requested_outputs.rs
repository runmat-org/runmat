use super::idivide_builtin;
use futures::executor::block_on;
use runmat_value::{IntValue, Value};

#[test]
fn requested_output_admission_preserves_discard_and_single_output() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count).expect("output-count scope");
        assert_eq!(
            block_on(idivide_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(IntValue::I16(6)),
                    Value::Int(IntValue::I16(2)),
                ])
            ))
            .expect("valid output count"),
            Value::Int(IntValue::I16(3)),
        );
    }
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = block_on(idivide_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Int(IntValue::I16(6)),
            Value::Int(IntValue::I16(2)),
        ]),
    ))
    .expect_err("output admission precedes operand consumption");
    assert_eq!(
        error.identifier(),
        runmat_builtins::IDIVIDE_ERROR_INVALID_INPUT.identifier
    );
}
