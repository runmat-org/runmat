use super::*;

#[test]
fn empty_known_callback_preserves_logical_output_class() {
    let empty = Tensor::new(Vec::new(), vec![0, 3]).expect("empty input");
    let comparison = Tensor::new(Vec::new(), vec![0, 3]).expect("empty comparison");
    let result = call(
        Value::FunctionHandle("gt".into()),
        vec![Value::Tensor(empty), Value::Tensor(comparison)],
    )
    .expect("arrayfun");
    let Value::LogicalArray(result) = result else {
        panic!("expected logical output");
    };
    assert_eq!(result.shape, vec![0, 3]);
    assert!(result.data.is_empty());
}

#[test]
fn empty_known_callback_preserves_integer_output_class() {
    let empty = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 2])
        .expect("empty integer input");
    let addend = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 2])
        .expect("empty integer addend");
    let result = call(
        Value::FunctionHandle("plus".into()),
        vec![Value::Tensor(empty), Value::Tensor(addend)],
    )
    .expect("arrayfun");
    let Value::Tensor(result) = result else {
        panic!("expected numeric output");
    };
    assert_eq!(result.shape, vec![0, 2]);
    assert_eq!(result.numeric_dtype(), runmat_value::NumericDType::U64);
    assert!(result.is_empty());
}
