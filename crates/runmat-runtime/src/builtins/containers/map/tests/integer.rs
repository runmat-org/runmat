use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn logical_scalar_tensor_conversion_preserves_low_rank_geometry() {
    for shape in [Vec::new(), vec![1]] {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), shape.clone()).unwrap();
        let Value::LogicalArray(output) =
            normalize_logical_value(Value::Tensor(input), BUILTIN_CONSTRUCTOR).unwrap()
        else {
            panic!("tensor conversion must retain logical array representation");
        };
        assert_eq!(output.shape, shape);
        assert_eq!(output.data, vec![1]);
    }
}

#[test]
fn typed_map_keys_preserve_full_uint64_range() {
    assert_eq!(
        unsigned_from_value(
            &Value::Int(IntValue::U64(u64::MAX)),
            u64::MAX,
            "key",
            BUILTIN_CONSTRUCTOR,
        )
        .expect("uint64 key"),
        u64::MAX
    );
    assert!(integer_from_value(
        &Value::Int(IntValue::U64(u64::MAX)),
        i64::MIN,
        i64::MAX,
        "key",
        BUILTIN_CONSTRUCTOR,
    )
    .is_err());
}

#[test]
fn scalar_map_helpers_read_typed_integer_tensor_storage_exactly() {
    let u64_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("uint64 key");
    assert_eq!(
        unsigned_from_value(
            &Value::Tensor(u64_tensor),
            u64::MAX,
            "key",
            BUILTIN_CONSTRUCTOR,
        )
        .expect("uint64 key"),
        u64::MAX
    );

    let i32_tensor =
        Tensor::new_integer(IntegerStorage::I32(vec![-7]), vec![1, 1]).expect("int32 key");
    assert_eq!(
        integer_from_value(
            &Value::Tensor(i32_tensor),
            i32::MIN as i64,
            i32::MAX as i64,
            "key",
            BUILTIN_CONSTRUCTOR,
        )
        .expect("int32 key"),
        -7
    );

    let logical_tensor =
        Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("logical key");
    assert!(
        bool_from_value(&Value::Tensor(logical_tensor), "key", BUILTIN_CONSTRUCTOR,)
            .expect("logical key")
    );
}

#[test]
fn vector_map_helpers_preserve_native_single_and_logicalize_typed_values() {
    let single = Tensor::from_f32(vec![1.25, 2.5], vec![1, 2]).expect("single values");
    let values = tensor_elements_to_values(&single);
    assert_eq!(values.len(), 2);
    for (value, expected) in values.into_iter().zip([1.25_f32, 2.5]) {
        let Value::Tensor(tensor) = value else {
            panic!("expected native-single scalar tensor");
        };
        assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
        assert_eq!(
            tensor.numeric_value_at(0),
            Some(runmat_value::NumericScalar::F32(expected))
        );
    }

    let logical = normalize_logical_value(
        Value::Tensor(
            Tensor::from_f32(vec![0.0, -0.0, 2.0, f32::NAN], vec![2, 2])
                .expect("single logical source"),
        ),
        BUILTIN_CONSTRUCTOR,
    );
    assert!(logical.is_err());
}

#[test]
fn vector_map_keys_and_values_read_typed_integer_storage_exactly() {
    let keys = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("uint64 keys");
    let values =
        Tensor::new_integer(IntegerStorage::I32(vec![11, 22]), vec![1, 2]).expect("values");

    let map =
        containers_map_builtin(vec![Value::Tensor(keys), Value::Tensor(values)]).expect("map");

    let key_payload = crate::make_cell(
        vec![
            Value::Int(IntValue::U64(u64::MAX - 1)),
            Value::Int(IntValue::U64(u64::MAX)),
        ],
        1,
        2,
    )
    .unwrap();
    let result = containers_map_values(map, vec![key_payload]).expect("selected values");
    match result {
        Value::Cell(cell) => {
            assert_eq!((cell.rows, cell.cols), (1, 2));
            assert_eq!(cell.data()[0], Value::Int(IntValue::I32(11)));
            assert_eq!(cell.data()[1], Value::Int(IntValue::I32(22)));
        }
        other => panic!("expected cell lookup result, got {:?}", other),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn map_numeric_admission_preserves_low_rank_and_exact_integer_values() {
    for shape in [Vec::new(), vec![1], vec![1, 1]] {
        let map = containers_map_builtin(vec![
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), shape.clone()).unwrap(),
            ),
            Value::Tensor(Tensor::new_integer(IntegerStorage::I32(vec![11]), shape).unwrap()),
        ])
        .expect("singleton key and value");
        let keys = crate::make_cell(vec![Value::Int(IntValue::U64(u64::MAX))], 1, 1).unwrap();
        let Value::Cell(values) = containers_map_values(map, vec![keys]).unwrap() else {
            panic!("expected selected map values");
        };
        assert_eq!(values.data(), vec![Value::Int(IntValue::I32(11))]);
    }
    for shape in [vec![2, 2], vec![2, 1, 2]] {
        let matrix = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], shape).unwrap());
        assert!(block_on(flatten_keys(&matrix, KeyType::Double, BUILTIN_CONSTRUCTOR)).is_err());
        assert!(block_on(flatten_values(&matrix, BUILTIN_CONSTRUCTOR)).is_err());
    }
}

#[test]
fn value_type_logical_reads_typed_integer_tensor_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
        .expect("integer logical source");
    let map = containers_map_builtin(vec![
        Value::from("KeyType"),
        Value::from("char"),
        Value::from("ValueType"),
        Value::from("logical"),
    ])
    .expect("map");
    let payload = crate::make_cell(vec![Value::from("mask")], 1, 1).unwrap();
    let error = containers_map_subsasgn(map, "()".to_string(), payload, Value::Tensor(tensor))
        .expect_err("declared logical ValueType accepts scalar values only");
    assert!(error.to_string().contains("logical scalars"));
}

fn all_integer_value_storages() -> Vec<(IntegerStorage, &'static str)> {
    vec![
        (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
        (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
        (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
        (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
        (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
        (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
        (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
        (IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]), "uint64"),
    ]
}

fn public_integer_key_storages() -> Vec<(IntegerStorage, &'static str)> {
    vec![
        (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
        (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
        (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
        (IntegerStorage::U64(vec![u64::MAX - 1, u64::MAX]), "uint64"),
    ]
}

fn property_text(map: Value, name: &str) -> String {
    let value = containers_map_subsref(map, ".".to_string(), Value::from(name)).unwrap();
    let Value::CharArray(chars) = value.value() else {
        panic!("expected char property");
    };
    runmat_value::decode_utf16_code_units(chars.data()).expect("valid UTF-16 fixture")
}

#[test]
fn constructor_subsref_and_values_preserve_all_integer_value_classes_exactly() {
    for (storage, class_name) in all_integer_value_storages() {
        let expected = storage.exact_values();
        let keys = crate::make_cell(vec![Value::from("a"), Value::from("b")], 1, 2).unwrap();
        let values = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
        let map = containers_map_builtin(vec![keys, values]).unwrap();
        assert_eq!(property_text(map.clone(), "ValueType"), class_name);

        let payload = crate::make_cell(vec![Value::from("b")], 1, 1).unwrap();
        assert_eq!(
            containers_map_subsref(map.clone(), "()".to_string(), payload).unwrap(),
            Value::Int(expected[1].clone())
        );

        let Value::Cell(all_values) = containers_map_values(map.clone(), Vec::new()).unwrap()
        else {
            panic!("expected values cell");
        };
        assert_eq!(
            all_values.data(),
            vec![
                Value::Int(expected[0].clone()),
                Value::Int(expected[1].clone())
            ]
        );

        let selected_keys =
            crate::make_cell(vec![Value::from("b"), Value::from("a")], 2, 1).unwrap();
        let Value::Cell(selected) =
            containers_map_values(map, vec![selected_keys]).expect("selected values")
        else {
            panic!("expected selected values cell");
        };
        assert_eq!((selected.rows, selected.cols), (2, 1));
        assert_eq!(
            selected.data(),
            vec![
                Value::Int(expected[1].clone()),
                Value::Int(expected[0].clone())
            ]
        );
    }
}

#[test]
fn keys_iskey_and_remove_preserve_all_public_integer_key_classes() {
    for (storage, class_name) in public_integer_key_storages() {
        let expected = storage.exact_values();
        let key_tensor = Value::Tensor(Tensor::new_integer(storage.clone(), vec![1, 2]).unwrap());
        let value_tensor = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap());
        let map = containers_map_builtin(vec![key_tensor.clone(), value_tensor]).unwrap();
        assert_eq!(property_text(map.clone(), "KeyType"), class_name);

        let Value::Cell(keys) = containers_map_keys(map.clone()).unwrap() else {
            panic!("expected keys cell");
        };
        assert_eq!(
            keys.data(),
            vec![
                Value::Int(expected[0].clone()),
                Value::Int(expected[1].clone())
            ]
        );

        let key_cell =
            crate::make_cell(expected.iter().cloned().map(Value::Int).collect(), 1, 2).unwrap();
        let Value::LogicalArray(found) =
            containers_map_is_key(map.clone(), key_cell).expect("isKey")
        else {
            panic!("expected logical array");
        };
        assert_eq!(found.shape, vec![1, 2]);
        assert_eq!(found.data, vec![1, 1]);

        let removed = containers_map_remove(map, Value::Int(expected[1].clone())).unwrap();
        assert_eq!(
            containers_map_is_key(removed, Value::Int(expected[1].clone())).unwrap(),
            Value::Bool(false)
        );
    }
}

#[test]
fn integer_key_methods_reject_class_mismatch_and_binary64_aliases() {
    let wide = 9_007_199_254_740_993_u64;
    let keys =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).unwrap());
    let map = containers_map_builtin(vec![keys, Value::Num(7.0)]).unwrap();
    let rounded_double = Value::Num(wide as f64);
    let payload = crate::make_cell(vec![rounded_double.clone()], 1, 1).unwrap();
    for error in [
        containers_map_subsref(map.clone(), "()".to_string(), payload.clone()).unwrap_err(),
        containers_map_is_key(map.clone(), rounded_double.clone()).unwrap_err(),
        containers_map_remove(map.clone(), rounded_double.clone()).unwrap_err(),
        containers_map_subsasgn(map.clone(), "()".to_string(), payload, Value::Num(9.0))
            .unwrap_err(),
    ] {
        assert!(error.to_string().contains("KeyType 'uint64'"));
    }
    let exact_payload = crate::make_cell(vec![Value::Int(IntValue::U64(wide))], 1, 1).unwrap();
    assert_eq!(
        containers_map_subsref(map, "()".to_string(), exact_payload).unwrap(),
        Value::Num(7.0)
    );
}

#[test]
fn direct_noncell_multi_key_method_inputs_reject() {
    let keys =
        Value::Tensor(Tensor::new_integer(IntegerStorage::I32(vec![1, 2]), vec![1, 2]).unwrap());
    let map = containers_map_builtin(vec![
        keys.clone(),
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    ])
    .unwrap();
    let is_key_error = containers_map_is_key(map.clone(), keys.clone()).unwrap_err();
    assert!(is_key_error.to_string().contains("cell array"));
    let remove_error = containers_map_remove(map, keys).unwrap_err();
    assert!(remove_error.to_string().contains("cell array"));
}

#[test]
fn subsasgn_explicit_integer_value_types_cover_all_eight_classes() {
    for (storage, class_name) in all_integer_value_storages() {
        let expected = storage.exact_values()[1].clone();
        let map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("uint64"),
            Value::from("ValueType"),
            Value::from(class_name),
        ])
        .unwrap();
        let payload = crate::make_cell(vec![Value::Int(IntValue::U64(u64::MAX))], 1, 1).unwrap();
        let updated = containers_map_subsasgn(
            map,
            "()".to_string(),
            payload.clone(),
            Value::Int(expected.clone()),
        )
        .unwrap();
        assert_eq!(
            updated
                .with_owned(|map| containers_map_subsref(map, "()".to_string(), payload))
                .unwrap(),
            Value::Int(expected)
        );
    }
}

#[test]
fn any_preserves_nonscalar_integer_arrays_and_declared_types_reject_them() {
    for (storage, class_name) in all_integer_value_storages() {
        let expected = storage.clone();
        let array = Value::Tensor(Tensor::new_integer(storage, vec![1, 2]).unwrap());
        let any_map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("char"),
            Value::from("ValueType"),
            Value::from("any"),
        ])
        .unwrap();
        let payload = crate::make_cell(vec![Value::from("array")], 1, 1).unwrap();
        let any_map =
            containers_map_subsasgn(any_map, "()".to_string(), payload.clone(), array.clone())
                .unwrap();
        let stored = any_map
            .with_owned(|map| containers_map_subsref(map, "()".to_string(), payload.clone()))
            .unwrap();
        let Value::Tensor(stored) = stored.value() else {
            panic!("expected exact integer tensor");
        };
        assert_eq!(stored.integer_storage(), Some(&expected));

        let typed_map = containers_map_builtin(vec![
            Value::from("KeyType"),
            Value::from("char"),
            Value::from("ValueType"),
            Value::from(class_name),
        ])
        .unwrap();
        assert!(containers_map_subsasgn(typed_map, "()".to_string(), payload, array,).is_err());
    }
}

#[test]
fn unsupported_narrow_integer_key_classes_reject_without_double_aliasing() {
    for storage in [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::U16(vec![1]),
    ] {
        let error = containers_map_builtin(vec![
            Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("int32"));
    }
}
