use super::super::*;
use futures::executor::block_on;

#[test]
fn bitwise_rejects_fractional_double() {
    let err = block_on(bitand_builtin(vec![Value::Num(1.5), Value::Num(1.0)]))
        .expect_err("fractional inputs should fail");
    assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn assumedtype_interprets_double_inputs_with_signed_bits_and_keeps_double_output() {
    let int8 = Value::String("int8".to_string());
    assert_eq!(
        block_on(bitand_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitand assumedtype"),
        Value::Num(2.0)
    );
    assert_eq!(
        block_on(bitor_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitor assumedtype"),
        Value::Num(-1.0)
    );
    assert_eq!(
        block_on(bitxor_builtin(vec![
            Value::Num(-5.0),
            Value::Num(6.0),
            int8.clone()
        ]))
        .expect("bitxor assumedtype"),
        Value::Num(-3.0)
    );
    assert_eq!(
        block_on(bitcmp_builtin(vec![Value::Num(-29.0), int8.clone()]))
            .expect("bitcmp assumedtype"),
        Value::Num(28.0)
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Num(-4.0),
            Value::Num(-1.0),
            int8.clone()
        ]))
        .expect("bitshift assumedtype"),
        Value::Num(-2.0)
    );
    assert_eq!(
        block_on(bitget_builtin(vec![
            Value::Num(-29.0),
            Value::Num(8.0),
            int8.clone()
        ]))
        .expect("bitget assumedtype"),
        Value::Num(1.0)
    );
    assert_eq!(
        block_on(bitset_builtin(vec![
            Value::Num(0.0),
            Value::Num(8.0),
            Value::Num(1.0),
            int8,
        ]))
        .expect("bitset assumedtype"),
        Value::Num(-128.0)
    );
}

#[test]
fn assumedtype_enforces_integer_classes_and_numeric_ranges() {
    let mismatch = block_on(bitxor_builtin(vec![
        Value::Int(IntValue::I8(1)),
        Value::Int(IntValue::I8(2)),
        Value::String("uint8".to_string()),
    ]))
    .expect_err("mismatched assumedtype");
    assert_eq!(mismatch.identifier(), ERROR_INVALID_INPUT.identifier);

    let out_of_range = block_on(bitset_builtin(vec![
        Value::Num(128.0),
        Value::Num(1.0),
        Value::String("int8".to_string()),
    ]))
    .expect_err("out of range assumedtype value");
    assert_eq!(out_of_range.identifier(), ERROR_INVALID_INPUT.identifier);

    let uint64_limit = block_on(bitcmp_builtin(vec![
        Value::Num(2_f64.powi(64)),
        Value::String("uint64".to_string()),
    ]))
    .expect_err("uint64 assumedtype excludes 2^64");
    assert_eq!(uint64_limit.identifier(), ERROR_INVALID_INPUT.identifier);
}

#[test]
fn assumedtype_preserves_typed_integer_output_and_dispatches_all_arities() {
    assert_eq!(
        block_on(bitand_builtin(vec![
            Value::Int(IntValue::I8(-5)),
            Value::Int(IntValue::I8(6)),
            Value::String("int8".to_string()),
        ]))
        .expect("typed bitand assumedtype"),
        Value::Int(IntValue::I8(2))
    );
    assert_eq!(
        crate::dispatcher::call_builtin(
            BITSET_NAME,
            &[
                Value::Num(0.0),
                Value::Num(8.0),
                Value::String("int8".to_string()),
            ],
        )
        .expect("bitset third assumedtype dispatch"),
        Value::Num(-128.0)
    );
}
