use super::*;
use runmat_value::CharArray;

#[test]
fn conj_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = CONJ_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"Y = conj(X)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_scalar_real() {
    let result = conj_builtin(Value::Num(-2.5)).expect("conj");
    match result {
        Value::Num(n) => assert!((n + 2.5).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_complex_scalar() {
    let result = conj_builtin(Value::Complex(3.0, 4.0)).expect("conj");
    match result {
        Value::Complex(re, im) => {
            assert!((re - 3.0).abs() < 1e-12);
            assert!((im + 4.0).abs() < 1e-12);
        }
        other => panic!("expected complex result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_complex_scalar_zero_imag_remains_complex() {
    let result = conj_builtin(Value::Complex(5.0, 0.0)).expect("conj");
    match result {
        Value::Complex(re, im) => {
            assert!((re - 5.0).abs() < 1e-12);
            assert_eq!(im, -0.0);
        }
        other => panic!("expected complex scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_preserves_logical_class() {
    let logical =
        LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).expect("logical array construction");
    let result = conj_builtin(Value::LogicalArray(logical)).expect("conj");
    match result {
        Value::LogicalArray(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            assert_eq!(t.data, vec![0, 1, 1, 0]);
        }
        other => panic!("expected logical result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_int_preserves_integer_class() {
    let result = conj_builtin(Value::Int(IntValue::I32(7))).expect("conj");
    match result {
        Value::Int(IntValue::I32(n)) => assert_eq!(n, 7),
        other => panic!("expected int32 scalar result, got {other:?}"),
    }
}

#[test]
fn conj_real_integer_arrays_preserve_all_eight_classes_and_wide_values() {
    let cases = [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ];
    for storage in cases {
        let input = Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap();
        let Value::Tensor(output) = conj_builtin(Value::Tensor(input)).expect("conj") else {
            panic!("expected tensor");
        };
        assert_eq!(output.integer_storage(), Some(&storage));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_complex_tensor_to_complex_tensor() {
    let tensor =
        ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).expect("complex tensor");
    let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape, vec![2, 1]);
            assert_eq!(ct.materialize_f64()[0], (1.0, -2.0));
            assert_eq!(ct.materialize_f64()[1], (-3.0, 4.0));
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_complex_tensor_zero_imag_remains_complex() {
    let tensor =
        ComplexTensor::new(vec![(1.0, 0.0), (2.0, -0.0)], vec![2, 1]).expect("complex tensor");
    let result = conj_builtin(Value::ComplexTensor(tensor)).expect("conj");
    match result {
        Value::ComplexTensor(t) => {
            assert_eq!(t.shape, vec![2, 1]);
            assert_eq!(t.materialize_f64(), vec![(1.0, -0.0), (2.0, 0.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn conj_complex_single_preserves_native_class_shape_and_empty_storage() {
    let tensor = ComplexTensor::from_f32(vec![(1.25, 2.5), (-3.0, -4.0)], vec![1, 2]).unwrap();
    let Value::ComplexTensor(output) = conj_builtin(Value::ComplexTensor(tensor)).expect("conj")
    else {
        panic!("expected complex single tensor");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.as_f32_slice().map(|values| values
            .iter()
            .copied()
            .map(<(f32, f32)>::from)
            .collect::<Vec<_>>()),
        Some(vec![(1.25, -2.5), (-3.0, 4.0)])
    );
    let empty = ComplexTensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let Value::ComplexTensor(output) = conj_builtin(Value::ComplexTensor(empty)).expect("conj")
    else {
        panic!("expected empty complex single tensor");
    };
    assert_eq!(output.shape, vec![0, 2]);
    assert_eq!(output.as_f32_slice(), Some(&[][..]));
}

#[test]
fn conj_typed_integer_imaginary_storage_preserves_class_and_saturates() {
    let cases = [
        (
            IntegerStorage::I8(vec![3, i8::MIN]),
            IntegerStorage::I8(vec![-3, i8::MAX]),
        ),
        (
            IntegerStorage::I16(vec![3, i16::MIN]),
            IntegerStorage::I16(vec![-3, i16::MAX]),
        ),
        (
            IntegerStorage::I32(vec![3, i32::MIN]),
            IntegerStorage::I32(vec![-3, i32::MAX]),
        ),
        (
            IntegerStorage::I64(vec![3, i64::MIN]),
            IntegerStorage::I64(vec![-3, i64::MAX]),
        ),
        (
            IntegerStorage::U8(vec![3, u8::MAX]),
            IntegerStorage::U8(vec![0, 0]),
        ),
        (
            IntegerStorage::U16(vec![3, u16::MAX]),
            IntegerStorage::U16(vec![0, 0]),
        ),
        (
            IntegerStorage::U32(vec![3, u32::MAX]),
            IntegerStorage::U32(vec![0, 0]),
        ),
        (
            IntegerStorage::U64(vec![3, u64::MAX]),
            IntegerStorage::U64(vec![0, 0]),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(conjugate_integer_imaginary_storage(input), expected);
    }
}

#[test]
fn conj_complex_integer_tensor_reads_storage_without_mirror() {
    let complex = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::I16(vec![-10, 20]),
            IntegerStorage::I16(vec![3, i16::MIN]),
        )
        .unwrap(),
        vec![1, 2],
    )
    .unwrap();

    let result = conj_builtin(Value::ComplexTensor(complex)).expect("conj");
    let Value::ComplexTensor(output) = result else {
        panic!("expected typed complex integer tensor");
    };
    assert_eq!(output.shape, vec![1, 2]);
    assert_eq!(
        output.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::I16(vec![-10, 20]),
                IntegerStorage::I16(vec![-3, i16::MAX]),
            )
            .unwrap()
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_char_array_returns_double_codes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let chars = CharArray::new("Hi".chars().collect(), 1, 2).expect("char array");
    let result = conj_builtin(Value::CharArray(chars)).expect("conj");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![1, 2]);
            assert_eq!(t.materialize_f64(), vec![72.0, 105.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn conj_char_extension_is_compatibility_gated() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = conj_builtin(Value::CharArray(CharArray::new_row("x"))).unwrap_err();
    assert_eq!(
        err.identifier(),
        CONJ_CHARACTER_INPUT_EXTENSION.error_identifier
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_errors_on_string_input() {
    let err = conj_builtin(Value::from("hello")).unwrap_err();
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected numeric input"));
    assert_eq!(identifier.as_deref(), CONJ_ERROR_INVALID_INPUT.identifier);
}
