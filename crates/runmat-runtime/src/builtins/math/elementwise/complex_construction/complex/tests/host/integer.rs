use super::super::*;

#[test]
fn complex_integer_components_preserve_uint64_storage_and_scalar_double_expansion() {
    let real = Tensor::new_integer(
        IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
        vec![1, 2],
    )
    .unwrap();
    let result = complex_call(Value::Tensor(real), vec![Value::Num(1.0)]).expect("complex");
    let Value::ComplexTensor(complex) = result else {
        panic!("integer complex values must retain complex tensor storage");
    };
    assert_eq!(complex.shape, vec![1, 2]);
    assert_eq!(
        complex.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_223_372_036_854_775_809, u64::MAX]),
                IntegerStorage::U64(vec![1, 1]),
            )
            .unwrap()
        )
    );
}

#[test]
fn complex_integer_components_broadcast_from_storage_without_mirrors() {
    let real = Tensor::new_integer(IntegerStorage::I32(vec![-3]), vec![1, 1]).expect("real scalar");
    let imag = Tensor::new_integer(IntegerStorage::I32(vec![7, -8, i32::MAX]), vec![3, 1])
        .expect("imag vector");

    let result = complex_call(Value::Tensor(real), vec![Value::Tensor(imag)])
        .expect("complex integer broadcast");
    let Value::ComplexTensor(output) = result else {
        panic!("expected typed complex integer tensor");
    };
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(
        output.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::I32(vec![-3, -3, -3]),
                IntegerStorage::I32(vec![7, -8, i32::MAX]),
            )
            .unwrap()
        )
    );
}

#[test]
fn complex_integer_scalar_keeps_exact_complex_storage() {
    let result = complex_call(
        Value::Int(IntValue::I64(i64::MIN)),
        vec![Value::Int(IntValue::I64(7))],
    )
    .expect("complex");
    let Value::ComplexTensor(complex) = result else {
        panic!("integer complex scalar must retain exact storage");
    };
    assert_eq!(complex.shape, vec![1, 1]);
    assert_eq!(
        complex.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN]),
                IntegerStorage::I64(vec![7]),
            )
            .unwrap()
        )
    );
}

#[test]
fn unary_complex_preserves_all_integer_classes() {
    let storages = vec![
        IntegerStorage::I8(vec![i8::MIN]),
        IntegerStorage::I16(vec![i16::MIN]),
        IntegerStorage::I32(vec![i32::MIN]),
        IntegerStorage::I64(vec![i64::MIN]),
        IntegerStorage::U8(vec![u8::MAX]),
        IntegerStorage::U16(vec![u16::MAX]),
        IntegerStorage::U32(vec![u32::MAX]),
        IntegerStorage::U64(vec![u64::MAX]),
    ];
    for storage in storages {
        let expected = storage.clone();
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer");
        let Value::ComplexTensor(result) =
            complex_call(Value::Tensor(tensor), Vec::new()).expect("complex")
        else {
            panic!("expected exact complex tensor");
        };
        let actual = result.integer_storage().expect("integer complex storage");
        assert_eq!(actual.real, expected);
        assert!(actual.imag.exact_values().iter().all(IntValue::is_zero));
    }
}

#[test]
fn binary_complex_preserves_all_integer_classes() {
    let storages = vec![
        IntegerStorage::I8(vec![-3]),
        IntegerStorage::I16(vec![-3]),
        IntegerStorage::I32(vec![-3]),
        IntegerStorage::I64(vec![-3]),
        IntegerStorage::U8(vec![3]),
        IntegerStorage::U16(vec![3]),
        IntegerStorage::U32(vec![3]),
        IntegerStorage::U64(vec![3]),
    ];
    for storage in storages {
        let expected = storage.clone();
        let real = Tensor::new_integer(storage.clone(), vec![1, 1]).expect("real");
        let imag = Tensor::new_integer(storage, vec![1, 1]).expect("imag");
        let Value::ComplexTensor(result) =
            complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex")
        else {
            panic!("expected exact complex tensor");
        };
        let actual = result.integer_storage().expect("integer complex storage");
        assert_eq!(actual.real, expected);
        assert_eq!(actual.imag, expected);
    }
}

#[test]
fn complex_rejects_mixed_integer_classes_and_non_scalar_double_arrays() {
    let mixed = complex_call(
        Value::Int(IntValue::I16(1)),
        vec![Value::Int(IntValue::U16(2))],
    )
    .expect_err("mixed integer classes should fail");
    assert_eq!(mixed.identifier(), COMPLEX_ERROR_INTEGER_CLASS.identifier);

    let doubles = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let array = complex_call(Value::Int(IntValue::I16(1)), vec![Value::Tensor(doubles)])
        .expect_err("integer inputs only permit scalar doubles as unlike operands");
    assert_eq!(array.identifier(), COMPLEX_ERROR_INTEGER_CLASS.identifier);
}

#[test]
fn complex_integer_accepts_full_scalar_double_tensor_peer() {
    let real = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("integer");
    let imag = Tensor::new(vec![3.0], vec![1, 1]).expect("full scalar double");
    let Value::ComplexTensor(result) =
        complex_call(Value::Tensor(real), vec![Value::Tensor(imag)]).expect("complex")
    else {
        panic!("expected integer complex");
    };
    assert_eq!(
        result.integer_storage().cloned(),
        Some(
            IntegerComplexStorage::new(
                IntegerStorage::I16(vec![1, 2]),
                IntegerStorage::I16(vec![3, 3])
            )
            .expect("storage")
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn complex_preserves_integer_inputs() {
    let result = complex_call(
        Value::Int(IntValue::I32(3)),
        vec![Value::Int(IntValue::I32(-4))],
    )
    .expect("complex");
    match result {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.shape, vec![1, 1]);
            assert_eq!(
                tensor.integer_storage().cloned(),
                Some(
                    IntegerComplexStorage::new(
                        IntegerStorage::I32(vec![3]),
                        IntegerStorage::I32(vec![-4]),
                    )
                    .expect("matching components")
                )
            );
        }
        other => panic!("expected typed complex integer result, got {other:?}"),
    }
}
