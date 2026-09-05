use super::*;

#[test]
fn angle_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = ANGLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"theta = angle(X)"));
    assert_eq!(ANGLE_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(
        ANGLE_INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::Rejected
    );
    assert!(ANGLE_INTEGER_CAPABILITIES[0].inputs[0].classes.is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_real_positive_negative() {
    let pos = angle_builtin(Value::Num(5.0)).expect("angle");
    assert_eq!(pos, Value::Num(0.0));

    let neg = angle_builtin(Value::Num(-3.0)).expect("angle");
    if let Value::Num(val) = neg {
        assert!((val - PI).abs() < 1e-12);
    } else {
        panic!("expected numeric result, got {neg:?}");
    }

    let zero = angle_builtin(Value::Num(0.0)).expect("angle");
    assert_eq!(zero, Value::Num(0.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_complex_scalar_matches_atan2() {
    let value = Value::Complex(3.0, -4.0);
    let result = angle_builtin(value).expect("angle");
    if let Value::Num(angle) = result {
        assert!((angle - (-4.0f64).atan2(3.0)).abs() < 1e-12);
    } else {
        panic!("expected numeric result");
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_tensor_values() {
    let tensor = Tensor::new(vec![1.0, -1.0, 0.0, 2.0], vec![2, 2]).unwrap();
    let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            assert!((out.materialize_f64()[0] - 0.0).abs() < 1e-12);
            assert!((out.materialize_f64()[1] - PI).abs() < 1e-12);
            assert_eq!(out.materialize_f64()[2], 0.0);
            assert_eq!(out.materialize_f64()[3], 0.0);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn angle_preserves_native_single_storage() {
    let tensor = Tensor::from_f32(vec![1.0, -1.0, 0.0], vec![1, 3]).unwrap();
    let result = angle_builtin(Value::Tensor(tensor)).expect("angle");
    match result {
        Value::Tensor(out) => match out.into_numeric_storage().expect("single storage") {
            NumericStorage::F32(values) => {
                assert_eq!(values[0], 0.0);
                assert!((values[1] - std::f32::consts::PI).abs() <= 2.0 * f32::EPSILON);
                assert_eq!(values[2], 0.0);
            }
            storage => panic!("expected single storage, got {storage:?}"),
        },
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn angle_rejects_every_real_integer_scalar_and_tensor_class() {
    for storage in all_integer_storages() {
        let class = storage.class_name();
        let scalar = storage.value_at(1).expect("integer scalar");
        let scalar_error =
            angle_builtin(Value::Int(scalar)).expect_err("integer scalar must reject");
        assert_eq!(
            scalar_error.identifier(),
            ANGLE_ERROR_INVALID_INPUT.identifier,
            "{class} scalar"
        );
        let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
        let tensor_error =
            angle_builtin(Value::Tensor(tensor)).expect_err("integer tensor must reject");
        assert_eq!(
            tensor_error.identifier(),
            ANGLE_ERROR_INVALID_INPUT.identifier,
            "{class} tensor"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_rejects_integer_logical_and_char_inputs() {
    let integer = angle_builtin(Value::Int(runmat_value::IntValue::I32(-1))).unwrap_err();
    assert_eq!(integer.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);

    let logical = LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
    let logical = angle_builtin(Value::LogicalArray(logical)).unwrap_err();
    assert_eq!(logical.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);

    let chars = CharArray::new("AB".chars().collect(), 1, 2).unwrap();
    let chars = angle_builtin(Value::CharArray(chars)).unwrap_err();
    assert_eq!(chars.identifier(), ANGLE_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn angle_rejects_every_typed_complex_integer_class() {
    for real in all_integer_storages() {
        let class = real.class_name();
        let imaginary = real.ones_like(real.len());
        let storage = IntegerComplexStorage::new(real, imaginary).expect("complex storage");
        let tensor = ComplexTensor::new_integer(storage, vec![1, 2]).expect("complex tensor");
        let error =
            angle_builtin(Value::ComplexTensor(tensor)).expect_err("complex integer must reject");
        assert_eq!(
            error.identifier(),
            ANGLE_ERROR_INVALID_INPUT.identifier,
            "{class} complex tensor"
        );
        assert!(error.message().contains("expected single or double"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_complex_tensor() {
    let data = vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
    let tensor = ComplexTensor::new(data, vec![2, 2]).unwrap();
    let result = angle_builtin(Value::ComplexTensor(tensor)).expect("angle");
    match result {
        Value::Tensor(out) => {
            let expected = [
                (1.0f64).atan2(1.0),
                (1.0f64).atan2(-1.0),
                (-1.0f64).atan2(-1.0),
                (-1.0f64).atan2(1.0),
            ];
            for (actual, target) in out.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - target).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_nan_propagates() {
    let result = angle_builtin(Value::Num(f64::NAN)).expect("angle");
    match result {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected numeric result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_rejects_strings() {
    let err = angle_builtin(Value::from("hello")).unwrap_err();
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected single or double input"));
    assert_eq!(identifier.as_deref(), ANGLE_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_rejects_string_arrays() {
    let array = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2]).unwrap();
    let err = angle_builtin(Value::StringArray(array)).unwrap_err();
    let identifier = err.identifier().map(str::to_string);
    assert!(err.message().contains("expected single or double input"));
    assert_eq!(identifier.as_deref(), ANGLE_ERROR_INVALID_INPUT.identifier);
}
