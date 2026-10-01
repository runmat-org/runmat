use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;
use runmat_value::Tensor;

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn tensor_out(value: Value) -> Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn scaled_false_restores_intercept_and_original_scale() {
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape, vec![2, 1]);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn ridge_accepts_typed_integer_response_design_and_k() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7]), 4, 1, f64::NAN);
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3]), 4, 1, f64::NAN);
    let k = poisoned_int_tensor(IntegerStorage::U8(vec![0, 1]), 1, 2, f64::NAN);
    let scaled = poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1, 1.0);
    let out = block_on(ridge_builtin(y, x, k, vec![scaled])).unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape, vec![2, 2]);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn scaled_true_returns_standardized_coefficients_for_each_k() {
    let y = tensor(vec![1.0, 3.0, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(
        y,
        x,
        tensor(vec![0.0, 1.0], 1, 2),
        Vec::new(),
    ))
    .unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape, vec![1, 2]);
    assert!(out.materialize_f64()[0] > out.materialize_f64()[1]);
    assert!((out.materialize_f64()[0] - 2.581_988_897).abs() < 1.0e-8);
}

#[test]
fn rows_with_nan_are_omitted() {
    let y = tensor(vec![1.0, f64::NAN, 5.0, 7.0], 4, 1);
    let x = tensor(vec![0.0, 1.0, 2.0, 3.0], 4, 1);
    let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
    let out = tensor_out(out);
    assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 2.0).abs() < 1.0e-10);
}

#[test]
fn invalid_inputs_are_rejected() {
    let y = tensor(vec![1.0, 2.0], 2, 1);
    let x = tensor(vec![1.0, 2.0], 2, 1);
    assert!(block_on(ridge_builtin(
        y.clone(),
        x.clone(),
        Value::Num(-1.0),
        Vec::new()
    ))
    .is_err());
    assert!(block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(2.0)])).is_err());
}

#[test]
fn wide_integer_scaled_flag_does_not_round_into_an_admitted_value() {
    let _compatibility = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
    for scaled in [
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        poisoned_int_tensor(IntegerStorage::U64(vec![u64::MAX]), 1, 1, 0.0),
    ] {
        assert!(block_on(ridge_builtin(
            y.clone(),
            x.clone(),
            Value::Num(0.0),
            vec![scaled]
        ))
        .is_err());
    }
}

#[test]
fn rank_deficient_k_zero_uses_minimum_norm_solution() {
    let y = tensor(vec![2.0, 4.0, 6.0], 3, 1);
    let x = tensor(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 3, 2);
    let out = block_on(ridge_builtin(y, x, Value::Num(0.0), vec![Value::Num(0.0)])).unwrap();
    let out = tensor_out(out);
    assert_eq!(out.shape, vec![3, 1]);
    assert!(out.materialize_f64()[0].abs() < 1.0e-10);
    assert!((out.materialize_f64()[1] - 1.0).abs() < 1.0e-10);
    assert!((out.materialize_f64()[2] - 0.5).abs() < 1.0e-10);
}
