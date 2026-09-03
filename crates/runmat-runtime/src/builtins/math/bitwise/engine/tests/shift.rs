use super::super::*;
use futures::executor::block_on;

#[test]
fn bitshift_supports_positive_and_negative_counts() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U32(3)),
            Value::Num(2.0)
        ]))
        .expect("left shift"),
        Value::Int(IntValue::U32(12))
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U32(8)),
            Value::Num(-1.0)
        ]))
        .expect("right shift"),
        Value::Int(IntValue::U32(4))
    );
}

#[test]
fn bitshift_preserves_integer_width() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U8(255)),
            Value::Num(1.0)
        ]))
        .expect("left shift"),
        Value::Int(IntValue::U8(254))
    );

    let tensor = Tensor::new_with_dtype(vec![255.0, 128.0], vec![1, 2], NumericDType::U8).unwrap();
    let out = block_on(bitshift_builtin(vec![
        Value::Tensor(tensor),
        Value::Num(1.0),
    ]))
    .expect("tensor shift");
    match out {
        Value::Tensor(t) => {
            assert_eq!(t.integer_storage(), Some(&IntegerStorage::U8(vec![254, 0])));
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn bitshift_sparse_double_preserves_sparse_implicit_zeros() {
    let sparse = runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![3.0, 5.0])
        .expect("sparse");
    let Value::SparseTensor(output) = block_on(bitshift_builtin(vec![
        Value::SparseTensor(sparse),
        Value::Num(1.0),
        Value::String("uint8".to_string()),
    ]))
    .expect("sparse bitshift") else {
        panic!("bitshift must preserve sparse storage when zero shifts to zero");
    };
    assert_eq!(output.shape(), vec![2, 2]);
    assert_eq!(output.get(0, 0), Some(6.0));
    assert_eq!(output.get(1, 1), Some(10.0));
    assert_eq!(output.nnz(), 2);
}

#[test]
fn bitshift_preserves_signed_arithmetic_and_64_bit_results() {
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::I64(-4)),
            Value::Int(IntValue::I64(-1)),
        ]))
        .expect("signed arithmetic right shift"),
        Value::Int(IntValue::I64(-2))
    );
    assert_eq!(
        block_on(bitshift_builtin(vec![
            Value::Int(IntValue::U64(1_u64 << 63)),
            Value::Int(IntValue::I64(-63)),
        ]))
        .expect("uint64 right shift"),
        Value::Int(IntValue::U64(1))
    );
}
