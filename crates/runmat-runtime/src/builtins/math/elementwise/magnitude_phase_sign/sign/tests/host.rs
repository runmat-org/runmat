use super::*;

#[test]
fn sign_preserves_native_single_storage() {
    let input = Tensor::from_f32(vec![-2.0, 0.0, 3.0, f32::NAN], vec![1, 4]).unwrap();
    let output = sign_tensor(input).unwrap();
    let NumericStorage::F32(values) = output.into_numeric_storage().unwrap() else {
        panic!("expected single storage");
    };
    assert_eq!(&values[..3], &[-1.0, 0.0, 1.0]);
    assert!(values[3].is_nan());
}

#[test]
fn sign_preserves_native_complex_single_storage() {
    let input = ComplexTensor::from_complex_storage(
        ComplexStorage::F32(vec![(3.0, 4.0), (0.0, 0.0), (f32::NAN, 1.0)].into()),
        vec![3, 1],
    )
    .unwrap();
    let Value::ComplexTensor(output) = sign_complex_tensor(input).unwrap() else {
        panic!("expected complex tensor");
    };
    let ComplexStorage::F32(values) = output.into_complex_storage() else {
        panic!("expected complex single storage");
    };
    assert!((values[0].0 - 0.6).abs() < 1e-6);
    assert!((values[0].1 - 0.8).abs() < 1e-6);
    assert_eq!(values[1], (0.0, 0.0).into());
    assert!(values[2].0.is_nan() && values[2].1.is_nan());
}

#[test]
fn sign_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = SIGN_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = sign(X)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_scalar_positive_negative_zero() {
    assert_eq!(sign_builtin(Value::Num(3.5)).unwrap(), Value::Num(1.0));
    assert_eq!(sign_builtin(Value::Num(-2.0)).unwrap(), Value::Num(-1.0));
    assert_eq!(sign_builtin(Value::Num(0.0)).unwrap(), Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_scalar_nan_propagates() {
    let result = sign_builtin(Value::Num(f64::NAN)).unwrap();
    match result {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected scalar NaN, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_tensor_mixed_values() {
    let tensor = Tensor::new(vec![-2.0, -0.0, 0.0, 5.0], vec![2, 2]).unwrap();
    let result = sign_builtin(Value::Tensor(tensor)).unwrap();
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            assert_eq!(out.materialize_f64(), vec![-1.0, 0.0, 0.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_complex_scalar_normalises() {
    let result = sign_builtin(Value::Complex(3.0, 4.0)).unwrap();
    match result {
        Value::Complex(re, im) => {
            assert!((re - 0.6).abs() < 1e-12);
            assert!((im - 0.8).abs() < 1e-12);
        }
        other => panic!("expected complex value, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_complex_tensor_handles_zero() {
    let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, -1.0)], vec![2, 1]).unwrap();
    let result = sign_builtin(Value::ComplexTensor(tensor)).unwrap();
    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![2, 1]);
            assert_eq!(out.materialize_f64()[0], (0.0, 0.0));
            let (re, im) = out.materialize_f64()[1];
            assert!((re - 0.7071067811865475).abs() < 1e-12);
            assert!((im + 0.7071067811865475).abs() < 1e-12);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_character_array() {
    let ca = CharArray::new("RunMat".chars().collect(), 1, 6).unwrap();
    let result = sign_builtin(Value::CharArray(ca)).unwrap();
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![1, 6]);
            assert!(out
                .materialize_f64()
                .iter()
                .all(|&v| (v - 1.0).abs() < 1e-12));
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_logical_array() {
    let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
    let result = sign_builtin(Value::LogicalArray(logical)).unwrap();
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}
