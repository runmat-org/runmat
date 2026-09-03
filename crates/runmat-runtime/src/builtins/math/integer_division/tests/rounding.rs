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
