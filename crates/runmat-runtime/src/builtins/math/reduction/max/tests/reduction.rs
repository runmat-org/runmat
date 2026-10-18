use super::*;
use runmat_value::{ComplexStorage, ComplexTensor, LogicalArray};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_scalar_returns_input() {
    let result = max_builtin(Value::Num(5.0), Vec::new()).expect("max");
    assert_eq!(result, Value::Num(5.0));
}

#[test]
fn max_requested_two_and_discarded_outputs_follow_one_evaluation() {
    let input = Tensor::new(vec![2.0, 5.0, 3.0], vec![3, 1]).expect("input");
    let both = block_on(crate::call_builtin_async_with_outputs(
        "max",
        &[Value::Tensor(input.clone())],
        2,
    ))
    .expect("two outputs");
    assert_eq!(
        both,
        Value::output_list(vec![Value::Num(5.0), Value::Num(2.0)])
    );
    let discarded = block_on(crate::call_builtin_async_with_outputs(
        "max",
        &[Value::Tensor(input)],
        0,
    ))
    .expect("zero outputs");
    assert_eq!(discarded, Value::output_list(Vec::new()));
}

#[test]
fn max_scalar_complex_zero_imaginary_stays_complex() {
    let (value, index) = evaluate(Value::Complex(5.0, 0.0), &[])
        .expect("max complex scalar")
        .into_pair();
    assert_eq!(value, Value::Complex(5.0, 0.0));
    assert_eq!(index, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_vector_with_indices() {
    let tensor = Tensor::new(vec![3.0, 1.0, 5.0], vec![3, 1]).unwrap();
    let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(5.0));
    assert_eq!(indices, Value::Num(3.0));
}

#[test]
fn max_reduction_preserves_native_single_storage() {
    let tensor = Tensor::from_f32(vec![3.0, 1.0, 4.0, 2.0], vec![2, 2]).unwrap();
    let (values, _) = evaluate(Value::Tensor(tensor), &[])
        .expect("max")
        .into_pair();
    let Value::Tensor(values) = values else {
        panic!("expected tensor values");
    };
    assert_eq!(
        values.into_numeric_storage().expect("single storage"),
        runmat_value::NumericStorage::F32(vec![3.0, 4.0])
    );
}

#[test]
fn max_logical_reduction_returns_double_values_and_indices() {
    let input = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical input");
    let (values, indices) = evaluate(Value::LogicalArray(input), &[])
        .expect("max logical")
        .into_pair();
    assert_eq!(
        values,
        Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap())
    );
    assert_eq!(
        indices,
        Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap())
    );
}

#[test]
fn max_empty_single_reduction_preserves_source_shape_and_value_class() {
    let input = Tensor::from_f32(Vec::new(), vec![0, 2]).expect("single empty");
    let (values, indices) = evaluate(Value::Tensor(input), &[])
        .expect("empty default")
        .into_pair();
    let Value::Tensor(values) = values else {
        panic!("single empty values")
    };
    assert_eq!(values.shape(), [0, 2]);
    assert_eq!(
        values.into_numeric_storage().expect("single storage"),
        runmat_value::NumericStorage::F32(Vec::new())
    );
    let Value::Tensor(indices) = indices else {
        panic!("empty indices")
    };
    assert_eq!(indices.shape(), [0, 2]);
    assert_eq!(
        indices.into_numeric_storage().expect("double storage"),
        runmat_value::NumericStorage::F64(Vec::new())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_row_vector_reduces_across_columns() {
    let tensor = Tensor::new(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
    let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(5.0));
    assert_eq!(indices, Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_single_row_vector_reduces_across_columns() {
    let tensor = Tensor::from_f32(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
    let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(
        values,
        Value::Tensor(Tensor::from_f32(vec![5.0], vec![1, 1]).unwrap())
    );
    assert_eq!(indices, Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_registered_single_row_vector_reduces_across_columns() {
    let tensor = Tensor::from_f32(vec![3.0, 1.0, 5.0], vec![1, 3]).unwrap();
    let value = block_on(crate::call_builtin_async_with_outputs(
        "max",
        &[Value::Tensor(tensor)],
        1,
    ))
    .expect("dispatch max");
    match value.value() {
        Value::OutputList(values) => {
            assert_eq!(values.len(), 1);
            assert_eq!(
                values[0],
                Value::Tensor(Tensor::from_f32(vec![5.0], vec![1, 1]).unwrap())
            );
        }
        other => assert_eq!(
            other,
            &Value::Tensor(Tensor::from_f32(vec![5.0], vec![1, 1]).unwrap())
        ),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_matrix_default_dimension() {
    let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let eval = evaluate(Value::Tensor(tensor), &[]).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [1, 3]);
            assert_eq!(t.materialize_f64(), vec![4.0, 2.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![2.0, 2.0, 2.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_reduction_complex_real_comparison() {
    let tensor = ComplexTensor::new(vec![(1.0, 2.0), (0.5, 5.0)], vec![2, 1]).expect("tensor");
    let args = vec![
        placeholder(),
        Value::from("ComparisonMethod"),
        Value::from("real"),
    ];
    let eval = evaluate(Value::ComplexTensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Complex(re, im) => {
            assert!((re - 1.0).abs() < 1e-12);
            assert!((im - 2.0).abs() < 1e-12);
        }
        other => panic!("expected complex scalar, got {other:?}"),
    }
    assert_eq!(indices, Value::Num(1.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_reduction_preserves_native_complex_single_storage() {
    let tensor = ComplexTensor::from_f32(vec![(1.0, 2.0), (0.5, 5.0)], vec![2, 1]).expect("tensor");
    let values = evaluate(Value::ComplexTensor(tensor), &[])
        .expect("evaluate")
        .into_value();
    let Value::ComplexTensor(values) = values else {
        panic!("expected typed complex scalar tensor");
    };
    assert_eq!(
        values.into_complex_storage(),
        ComplexStorage::F32(vec![(0.5, 5.0)].into())
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_complex_real_comparison() {
    let lhs = ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap();
    let rhs = ComplexTensor::new(vec![(0.5, 5.0)], vec![1, 1]).unwrap();
    let args = vec![
        Value::ComplexTensor(rhs),
        Value::from("ComparisonMethod"),
        Value::from("real"),
    ];
    let eval = evaluate(Value::ComplexTensor(lhs), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Complex(1.0, 2.0));
    assert_eq!(indices, Value::Num(1.0));
}
