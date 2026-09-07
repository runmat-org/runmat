use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_product_matches_expected() {
    let a = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::new(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0], vec![3, 2]).unwrap();
    let result = mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).expect("mtimes");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![58.0, 139.0, 64.0, 154.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn mtimes_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = MTIMES_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert!(labels.contains(&"C = mtimes(A, B)"));
}

#[test]
fn mtimes_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = MTIMES_DESCRIPTOR
        .errors
        .iter()
        .map(|err| err.code)
        .collect();
    assert!(codes.contains(&"RM.MTIMES.INVALID_INPUT"));
    assert!(codes.contains(&"RM.MTIMES.INTERNAL"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn scalar_matrix_product() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = mtimes_builtin(Value::Num(0.5), Value::Tensor(a)).expect("mtimes");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![0.5, 1.0, 1.5, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_scalar_product() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = mtimes_builtin(Value::Tensor(a), Value::Num(3.0)).expect("mtimes");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.materialize_f64(), vec![3.0, 6.0, 9.0, 12.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dot_product_returns_scalar() {
    let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let col = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
    let result = mtimes_builtin(Value::Tensor(row), Value::Tensor(col)).expect("mtimes");
    match result {
        Value::Num(value) => assert!((value - 32.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_matrix_product() {
    let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
    let matrix = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let result =
        mtimes_builtin(Value::LogicalArray(logical), Value::Tensor(matrix)).expect("mtimes");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.materialize_f64(), vec![2.0, 3.0, 4.0, 5.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_tensor_product() {
    let ct = runmat_value::ComplexTensor::new(
        vec![(1.0, 2.0), (3.0, -4.0), (5.0, 6.0), (7.0, -8.0)],
        vec![2, 2],
    )
    .unwrap();
    let scalar = Value::Complex(1.0, -1.0);
    let result = mtimes_builtin(Value::ComplexTensor(ct.clone()), scalar).expect("mtimes");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, ct.shape);
            assert_eq!(
                t.materialize_f64(),
                vec![(3.0, 1.0), (-1.0, -7.0), (11.0, 1.0), (-1.0, -15.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn inner_dimension_mismatch_errors() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 6.0, 7.0], vec![3, 1]).unwrap();
    let err = unwrap_error(mtimes_builtin(Value::Tensor(a), Value::Tensor(b)).unwrap_err());
    assert_eq!(err.identifier(), MTIMES_ERROR_INVALID_INPUT.identifier);
    assert!(
        err.message().contains("Inner matrix dimensions must agree"),
        "unexpected error message: {err}"
    );
}
