use super::*;
use runmat_builtins::SQRT_DESCRIPTOR;
use runmat_value::{
    CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, NumericStorage, Tensor,
};

#[test]
fn sqrt_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = SQRT_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = sqrt(X)"));
}

#[test]
fn sqrt_string_rejected_with_stable_identifier() {
    let err = sqrt_builtin(Value::from("bad")).expect_err("expected input error");
    assert_eq!(err.identifier(), SQRT_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_scalar_positive() {
    let result = sqrt_builtin(Value::Num(9.0)).expect("sqrt");
    match result {
        Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[test]
fn sqrt_canonicalizes_negative_zero() {
    let result = sqrt_builtin(Value::Num(-0.0)).expect("sqrt");
    let Value::Num(value) = result else {
        panic!("expected scalar result, got {result:?}");
    };
    assert_eq!(value.to_bits(), 0.0f64.to_bits());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_scalar_negative() {
    let result = sqrt_builtin(Value::Num(-4.0)).expect("sqrt");
    match result {
        Value::Complex(re, im) => {
            assert!(re.abs() < 1e-12);
            assert!((im - 2.0).abs() < 1e-12);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_bool_true() {
    let result = sqrt_builtin(Value::Bool(true)).expect("sqrt");
    match result {
        Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_logical_array_inputs() {
    let logical = LogicalArray::new(vec![1u8, 0, 1, 0], vec![2, 2]).expect("logical");
    let result = sqrt_builtin(Value::LogicalArray(logical)).expect("sqrt");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
            assert!(t.materialize_f64()[1].abs() < 1e-12);
            assert!((t.materialize_f64()[2] - 1.0).abs() < 1e-12);
            assert!(t.materialize_f64()[3].abs() < 1e-12);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_tensor_with_negatives() {
    let tensor = Tensor::new(vec![-1.0, 4.0], vec![1, 2]).unwrap();
    let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![1, 2]);
            assert!(ct.materialize_f64()[0].0.abs() < 1e-12);
            assert!((ct.materialize_f64()[0].1 - 1.0).abs() < 1e-12);
            assert!((ct.materialize_f64()[1].0 - 2.0).abs() < 1e-12);
            assert!(ct.materialize_f64()[1].1.abs() < 1e-12);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_char_array_inputs() {
    let chars = CharArray::new("AZ".chars().collect(), 1, 2).unwrap();
    let result = sqrt_builtin(Value::CharArray(chars)).expect("sqrt");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert!((t.materialize_f64()[0] - (65.0f64).sqrt()).abs() < 1e-12);
            assert!((t.materialize_f64()[1] - (90.0f64).sqrt()).abs() < 1e-12);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_string_input_errors() {
    let err = sqrt_builtin(Value::from("hello")).unwrap_err();
    assert_eq!(err.identifier(), SQRT_ERROR_INVALID_INPUT.identifier);
    assert!(err.message().contains("expected numeric input"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_complex_scalar() {
    let result = sqrt_builtin(Value::Complex(3.0, 4.0)).expect("sqrt");
    match result {
        Value::Complex(re, im) => {
            assert!((re - 2.0).abs() < 1e-12);
            assert!((im - 1.0).abs() < 1e-12);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_integer_argument() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = sqrt_builtin(Value::Int(IntValue::I32(9))).expect("sqrt");
    match result {
        Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_reads_typed_integer_tensor_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor = Tensor::new_integer(IntegerStorage::U32(vec![0, 4, 9]), vec![3, 1])
        .expect("integer tensor");

    let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![3, 1]);
            assert_eq!(out.materialize_f64(), vec![0.0, 2.0, 3.0]);
            assert!(out.integer_storage().is_none());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sqrt_negative_typed_integer_tensor_promotes_to_complex_from_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let tensor =
        Tensor::new_integer(IntegerStorage::I32(vec![-4, 9]), vec![1, 2]).expect("integer tensor");

    let result = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt");
    match result {
        Value::ComplexTensor(out) => {
            assert_eq!(out.shape, vec![1, 2]);
            assert_eq!(out.materialize_f64()[0], (0.0, 2.0));
            assert_eq!(out.materialize_f64()[1], (3.0, 0.0));
        }
        other => panic!("expected complex tensor result, got {other:?}"),
    }
}

#[test]
fn sqrt_preserves_native_single_real_complex_negative_and_empty_storage() {
    let tensor = Tensor::from_f32(vec![0.0, 4.0], vec![2, 1]).unwrap();
    let Value::Tensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt") else {
        panic!("expected single real tensor");
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![0.0, 2.0])
    );

    let tensor = Tensor::from_f32(vec![-4.0, 9.0], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::Tensor(tensor)).expect("sqrt") else {
        panic!("expected complex single tensor");
    };
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(0.0, 2.0), (3.0, 0.0)])
    );
    let complex = ComplexTensor::from_f32(vec![(3.0, 4.0)], vec![1, 1]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::ComplexTensor(complex)).expect("sqrt")
    else {
        panic!("one-element complex single must retain class");
    };
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![sqrt_complex_parts_f32(3.0, 4.0)])
    );
    let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let Value::ComplexTensor(output) = sqrt_builtin(Value::ComplexTensor(empty)).expect("sqrt")
    else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(output.shape, vec![0, 2]);
    assert_eq!(output.as_f32_slice(), Some(&[][..]));
}
