use super::*;
use runmat_value::IntValue;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_all_linear_index() {
    let tensor = Tensor::new((1..=12).map(|v| v as f64).collect::<Vec<_>>(), vec![3, 4]).unwrap();
    let args = vec![placeholder(), Value::from("all")];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(12.0));
    assert_eq!(indices, Value::Num(12.0));

    let args_linear = vec![placeholder(), Value::from("linear")];
    let eval = evaluate(
        Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
        &args_linear,
    )
    .expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(3.0));
    assert_eq!(indices, Value::Num(2.0));
}

#[test]
fn empty_selector_then_all_rejects_but_placeholder_then_all_is_valid() {
    let input = Tensor::new(vec![2.0, 3.0], vec![2, 1]).expect("input");
    let valid = evaluate(
        Value::Tensor(input.clone()),
        &[placeholder(), Value::from("all")],
    )
    .expect("placeholder then all");
    assert_eq!(valid.into_value(), Value::Num(3.0));
    let error = evaluate(
        Value::Tensor(input),
        &[placeholder(), placeholder(), Value::from("all")],
    )
    .expect_err("empty selector then all");
    assert_eq!(error.identifier(), MAX_ERROR_INVALID_ARGUMENT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_with_omitnan() {
    let tensor = Tensor::new(vec![f64::NAN, 4.0, 2.0], vec![3, 1]).unwrap();
    let args = vec![placeholder(), Value::from("omitnan")];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(4.0));
    assert_eq!(indices, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_omitnan_all_nan_slice() {
    let tensor = Tensor::new(vec![f64::NAN, f64::NAN], vec![2, 1]).unwrap();
    let args = vec![placeholder(), Value::from("omitnan")];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected scalar NaN, got {other:?}"),
    }
    match indices {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected scalar NaN index, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_reduction_abs_comparison() {
    let tensor = Tensor::new(vec![1.0, -3.0, -2.0, 4.0], vec![2, 2]).unwrap();
    let args = vec![
        placeholder(),
        Value::from("ComparisonMethod"),
        Value::from("abs"),
    ];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [1, 2]);
            assert_eq!(t.materialize_f64(), vec![-3.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![2.0, 2.0]);
        }
        other => panic!("expected tensor indices, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_dimension_argument_parsing() {
    let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let dims = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let args = vec![placeholder(), Value::Tensor(dims)];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(4.0));
    assert_eq!(indices, Value::Num(2.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_vecdim_duplicate_entries() {
    let tensor = Tensor::new(vec![5.0, 2.0, 7.0, 1.0], vec![2, 2]).unwrap();
    let dims = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).unwrap();
    let args = vec![placeholder(), Value::Tensor(dims)];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    assert_eq!(values, Value::Num(7.0));
    assert_eq!(indices, Value::Num(3.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_dimension_gpu_argument_errors() {
    let tensor = Tensor::new(vec![3.0, 1.0], vec![2, 1]).unwrap();
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident reduction admission fixture");
    let dim_handle = Value::GpuTensor(allocation.handle.clone());
    let err =
        evaluate(Value::Tensor(tensor), &[placeholder(), dim_handle]).expect_err("expected error");
    assert_eq!(err.identifier(), MAX_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err
        .message()
        .contains("dimension arguments must reside on the host"));
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
    drop(err);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_invalid_comparison_method_errors() {
    let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let args = vec![
        placeholder(),
        Value::from("ComparisonMethod"),
        Value::from("chebyshev"),
    ];
    let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
    assert_eq!(err.identifier(), MAX_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("unsupported ComparisonMethod"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_dimension_numeric_argument() {
    let tensor = Tensor::new(vec![3.0, 4.0, 1.0, 2.0], vec![2, 2]).unwrap();
    let args = vec![placeholder(), Value::Num(2.0)];
    let eval = evaluate(Value::Tensor(tensor), &args).expect("evaluate");
    let (values, indices) = eval.into_pair();
    match values {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), [2, 1]);
            assert_eq!(t.materialize_f64(), vec![3.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    match indices {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![1.0, 1.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn max_rejects_invalid_dimension() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let args = vec![placeholder(), Value::Int(IntValue::I32(0))];
    let err = evaluate(Value::Tensor(tensor), &args).expect_err("expected error");
    assert_eq!(err.identifier(), MAX_ERROR_INVALID_ARGUMENT.identifier);
    assert!(err.message().contains("dimension must be >= 1"));
}
