use super::*;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage};

mod provider_preflight;

fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn mirrorless_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn complex_tensor(data: Vec<(f64, f64)>, rows: usize, cols: usize) -> Value {
    Value::ComplexTensor(ComplexTensor::new(data, vec![rows, cols]).unwrap())
}

#[test]
fn third_position_algorithm_has_a_typed_admission_route() {
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let output = block_on(lscov_builtin(a, b, vec![Value::from("orth")])).unwrap();
    let x = match output {
        Value::OutputList(values) => values.into_iter().next().unwrap(),
        value => value,
    };
    let Value::Tensor(x) = x else {
        panic!("expected numeric coefficients")
    };
    assert_eq!(x.shape, vec![2, 1]);
    assert!(x
        .materialize_f64()
        .iter()
        .zip([1.0, 2.0])
        .all(|(actual, wanted)| (*actual - wanted).abs() < 1e-8));
}

#[test]
fn rank_one_response_matches_tensor_row_vector_admission() {
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0], vec![3]).unwrap());
    let output = block_on(lscov_builtin(a, b, Vec::new())).unwrap();
    let Value::Tensor(x) = output else {
        panic!("expected numeric coefficients")
    };
    assert_eq!(x.shape, vec![2, 1]);
    assert!((x.materialize_f64()[0] - 1.0).abs() < 1e-8);
    assert!((x.materialize_f64()[1] - 2.0).abs() < 1e-8);
}

#[test]
fn empty_nd_weight_is_the_unweighted_fit() {
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 2, 3]).unwrap());
    let baseline = block_on(lscov_builtin(a.clone(), b.clone(), Vec::new())).unwrap();
    let weighted = block_on(lscov_builtin(a, b, vec![empty])).unwrap();
    assert_eq!(baseline, weighted);
}

fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values,
        other => panic!("expected output list, got {other:?}"),
    }
}

fn tensor_ref(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn complex_ref(value: &Value) -> &ComplexTensor {
    match value {
        Value::ComplexTensor(tensor) => tensor,
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

fn numeric_matrix_shape(value: &Value) -> Vec<usize> {
    match value {
        Value::Tensor(tensor) => tensor.shape.clone(),
        Value::ComplexTensor(tensor) => tensor.shape.clone(),
        other => panic!("expected numeric matrix, got {other:?}"),
    }
}

fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1.0e-9,
        "{left:?} not close to {right:?}"
    );
}

#[test]
fn lscov_solves_ordinary_least_squares_outputs() {
    let _guard = crate::output_count::push_output_count(Some(4));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert_close(x.materialize_f64()[0], 1.0);
    assert_close(x.materialize_f64()[1], 2.0);
    assert_eq!(tensor_ref(&out[1]).shape, vec![2, 1]);
    assert_close(tensor_ref(&out[2]).materialize_f64()[0], 0.0);
    assert_eq!(tensor_ref(&out[3]).shape, vec![2, 2]);
}

#[test]
fn lscov_accepts_typed_integer_matrices_and_weights() {
    let _guard = crate::output_count::push_output_count(Some(1));
    let a = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 0, 1, 2]), 3, 2, f64::NAN);
    let b = poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5]), 3, 1, f64::NAN);
    let weights = poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 1]), 3, 1, f64::NAN);
    let out = outputs(block_on(lscov_builtin(a, b, vec![weights])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert_close(x.materialize_f64()[0], 1.0);
    assert_close(x.materialize_f64()[1], 2.0);
}

#[test]
fn lscov_reads_typed_integer_covariance_from_exact_storage() {
    let _guard = crate::output_count::push_output_count(Some(1));
    let a = poisoned_int_tensor(IntegerStorage::I16(vec![1, 1, 1, 0, 1, 2]), 3, 2, f64::NAN);
    let b = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 10]), 3, 1, f64::NAN);
    let v = poisoned_int_tensor(
        IntegerStorage::U16(vec![
            1, 0, 0, //
            0, 1, 0, //
            0, 0, 100,
        ]),
        3,
        3,
        f64::NAN,
    );
    let alg = Value::CharArray(CharArray::new_row("chol"));
    let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert!(x.materialize_f64().iter().all(|value| value.is_finite()));
}

#[test]
fn lscov_weighting_length_uses_typed_integer_storage_not_mirror() {
    let weighting = block_on(parse_weighting(
        mirrorless_int_tensor(IntegerStorage::U16(vec![1, 2, 3]), 3, 1),
        3,
    ))
    .unwrap();
    match weighting {
        Weighting::Weights(values) => assert_eq!(values, vec![1.0, 2.0, 3.0]),
        other => panic!("expected vector weighting, got {other:?}"),
    }
}

#[test]
fn lscov_supports_weight_vector_weighting() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 2.0, 10.0], 3, 1);
    let v = tensor(vec![1.0, 1.0, 100.0], 3, 1);
    let out = outputs(block_on(lscov_builtin(a, b, vec![v])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_close(x.materialize_f64()[0], -0.3972055888223553);
    assert_close(x.materialize_f64()[1], 5.191616766467066);
    let stdx = tensor_ref(&out[1]);
    assert_eq!(stdx.shape, vec![2, 1]);
    assert!(stdx.materialize_f64().iter().all(|value| value.is_finite()));
    assert!(tensor_ref(&out[2]).materialize_f64()[0].is_finite());
}

#[test]
fn lscov_supports_zero_weight_observations() {
    let _guard = crate::output_count::push_output_count(Some(1));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 100.0], 3, 1);
    let w = tensor(vec![1.0, 1.0, 0.0], 3, 1);
    let out = outputs(block_on(lscov_builtin(a, b, vec![w])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert_close(x.materialize_f64()[0], 1.0);
    assert_close(x.materialize_f64()[1], 2.0);
}

#[test]
fn lscov_supports_full_covariance_and_chol_algorithm() {
    let _guard = crate::output_count::push_output_count(Some(1));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 2.0, 10.0], 3, 1);
    let v = tensor(
        vec![
            1.0, 0.1, 0.0, //
            0.1, 1.0, 0.0, //
            0.0, 0.0, 100.0,
        ],
        3,
        3,
    );
    let alg = Value::CharArray(CharArray::new_row("chol"));
    let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert!(x.materialize_f64().iter().all(|value| value.is_finite()));
}

#[test]
fn lscov_orth_handles_singular_psd_covariance() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 100.0], 3, 1);
    let v = tensor(
        vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        3,
        3,
    );
    let alg = Value::String("orth".to_string());
    let out = outputs(block_on(lscov_builtin(a, b, vec![v, alg])).unwrap());
    let x = tensor_ref(&out[0]);
    assert_close(x.materialize_f64()[0], 1.0);
    assert_close(x.materialize_f64()[1], 2.0);
    assert_close(tensor_ref(&out[2]).materialize_f64()[0], 0.0);
}

#[test]
fn lscov_supports_matrix_rhs() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2);
    let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
    let x = tensor_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 2]);
    assert_close(x.materialize_f64()[0], 1.0);
    assert_close(x.materialize_f64()[1], 2.0);
    assert_close(x.materialize_f64()[2], 2.0);
    assert_close(x.materialize_f64()[3], 2.0);
    assert_eq!(tensor_ref(&out[2]).shape, vec![1, 2]);
}

#[test]
fn lscov_supports_complex_design_and_response() {
    let _guard = crate::output_count::push_output_count(Some(4));
    let a = complex_tensor(
        vec![
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, -1.0),
        ],
        3,
        2,
    );
    let b = complex_tensor(vec![(1.0, 1.0), (3.0, 0.0), (5.0, -1.0)], 3, 1);
    let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
    let x = complex_ref(&out[0]);
    assert_eq!(x.shape, vec![2, 1]);
    assert!(x
        .materialize_f64()
        .iter()
        .all(|(re, im)| re.is_finite() && im.is_finite()));
    assert_eq!(tensor_ref(&out[1]).shape, vec![2, 1]);
    assert_eq!(tensor_ref(&out[2]).shape, vec![1, 1]);
    assert_eq!(numeric_matrix_shape(&out[3]), vec![2, 2]);
}

#[test]
fn lscov_rejects_fourth_output_for_matrix_rhs() {
    let _guard = crate::output_count::push_output_count(Some(4));
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 3, 2);
    let err = block_on(lscov_builtin(a, b, Vec::new())).unwrap_err();
    assert!(err.message().contains("fourth output S"));
}

#[test]
fn lscov_accepts_zero_weights() {
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let v = tensor(vec![1.0, 0.0, 1.0], 3, 1);
    block_on(lscov_builtin(a, b, vec![v])).expect("zero weights are allowed");
}

#[test]
fn lscov_rejects_negative_weights() {
    let a = tensor(vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 2);
    let b = tensor(vec![1.0, 3.0, 5.0], 3, 1);
    let v = tensor(vec![1.0, -1.0, 1.0], 3, 1);
    let err = block_on(lscov_builtin(a, b, vec![v])).unwrap_err();
    assert!(err.message().contains("finite nonnegative"));
}

#[test]
fn lscov_underdetermined_mse_is_zero() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let a = tensor(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 2, 3);
    let b = tensor(vec![1.0, 2.0], 2, 1);
    let out = outputs(block_on(lscov_builtin(a, b, Vec::new())).unwrap());
    assert_eq!(tensor_ref(&out[0]).shape, vec![3, 1]);
    assert_close(tensor_ref(&out[2]).materialize_f64()[0], 0.0);
}
