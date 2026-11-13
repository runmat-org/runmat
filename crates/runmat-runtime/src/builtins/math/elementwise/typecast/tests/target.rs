use super::super::typecast_builtin;
use futures::executor::block_on;
use runmat_value::{IntValue, Value};

#[test]
fn class_selector_is_canonical_and_case_insensitive_at_the_boundary() {
    let Value::Tensor(output) = block_on(typecast_builtin(vec![
        Value::Int(IntValue::U16(0x1234)),
        Value::String("UINT8".into()),
    ]))
    .expect("case-insensitive class") else {
        panic!("expected tensor");
    };
    assert_eq!(output.len(), 2);
}

#[test]
fn invalid_classes_and_nonliteral_like_syntax_are_rejected() {
    for selector in ["char", "table", "uint128"] {
        assert!(block_on(typecast_builtin(vec![
            Value::Int(IntValue::U16(1)),
            Value::String(selector.into()),
        ]))
        .is_err());
    }
    assert!(block_on(typecast_builtin(vec![
        Value::Int(IntValue::U16(1)),
        Value::String("uint8".into()),
        Value::Int(IntValue::U8(0)),
    ]))
    .is_err());
}
