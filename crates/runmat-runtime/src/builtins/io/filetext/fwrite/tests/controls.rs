use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"count = fwrite(fid, data)"));
    assert!(labels.contains(&"count = fwrite(fid, data, precision, skip)"));
    assert!(labels.contains(&"count = fwrite(fid, data, precision, machinefmt)"));
    assert!(labels.contains(&"count = fwrite(fid, data, precision, skip, machinefmt)"));
}

#[test]
fn fwrite_integer_capabilities_and_control_roles_are_independently_gated() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 3);
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let fid = preflight_control(&Value::Int(IntValue::I32(3)), ControlRole::FileId).unwrap_err();
    assert_eq!(
        fid.identifier(),
        Some("RunMat:compatibility:FwriteIntegerIdExtension")
    );
    let skip = preflight_control(&Value::Int(IntValue::U16(2)), ControlRole::Skip).unwrap_err();
    assert_eq!(
        skip.identifier(),
        Some("RunMat:compatibility:FwriteIntegerSkipExtension")
    );
}

#[test]
fn fwrite_classifies_resident_skip_before_gathering() {
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let args = vec![Value::from("uint8"), resident];
    let (_, skip, _) = classify_arguments(crate::call::FunctionArgumentView::from_slice(&args))
        .expect("classified controls");
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = preflight_control(skip.expect("skip"), ControlRole::Skip).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FwriteResidentControlExtension")
    );
}

#[test]
fn fwrite_nan_to_integer_precision_writes_zero() {
    assert_eq!(to_u8(f64::NAN), 0);
    assert_eq!(
        encode_i16(f64::NAN, Endianness::Little),
        0_i16.to_le_bytes()
    );
    assert_eq!(encode_u64(f64::NAN, Endianness::Big), 0_u64.to_be_bytes());
}

#[test]
fn typed_integer_precision_conversions_stay_in_integer_domain() {
    let signed_min = WriteElement::Integer(IntValue::I64(i64::MIN));
    let unsigned_max = WriteElement::Integer(IntValue::U64(u64::MAX));
    let wide_unsigned = WriteElement::Integer(IntValue::U64(9_007_199_254_740_993));

    assert_eq!(integer_unsigned(&signed_min, u64::MAX), Some(0));
    assert_eq!(
        integer_signed(&unsigned_max, i64::MIN, i64::MAX),
        Some(i64::MAX)
    );
    assert_eq!(
        integer_unsigned(&wide_unsigned, u32::MAX as u64),
        Some(u32::MAX as u64)
    );
    assert_eq!(
        integer_signed(&wide_unsigned, i32::MIN as i64, i32::MAX as i64),
        Some(i32::MAX as i64)
    );
}

#[test]
fn fwrite_borrows_each_integer_storage_class_without_reading_f64_mirror() {
    let cases = [
        (IntegerStorage::I8(vec![-8]), -8_i128),
        (IntegerStorage::I16(vec![-16]), -16),
        (IntegerStorage::I32(vec![-32]), -32),
        (IntegerStorage::I64(vec![i64::MIN]), i64::MIN as i128),
        (IntegerStorage::U8(vec![8]), 8),
        (IntegerStorage::U16(vec![16]), 16),
        (IntegerStorage::U32(vec![32]), 32),
        (IntegerStorage::U64(vec![u64::MAX]), u64::MAX as i128),
    ];

    for (storage, expected) in cases {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed tensor");
        let value = Value::Tensor(tensor);
        let elements: Vec<_> = borrow_elements(&value).expect("typed elements").collect();
        assert_eq!(elements.len(), 1);
        assert_eq!(integer_raw(&elements[0]), Some(expected));
    }
}

#[test]
fn fwrite_scalar_parser_reads_typed_integer_storage_exactly() {
    let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
    assert_eq!(
        numeric_scalar(&Value::Tensor(scalar), "scalar").expect("scalar"),
        7.0
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn fwrite_skip_parses_integer_values_exactly() {
    let exact = (1_u64 << 53) + 1;

    assert_eq!(
        parse_skip(Some(&Value::Int(IntValue::U64(exact)))).unwrap(),
        exact as usize
    );
    assert!(parse_skip(Some(&Value::Int(IntValue::U64(u64::MAX)))).is_err());
    assert!(parse_skip(Some(&Value::Int(IntValue::I8(-1)))).is_err());
    assert!(parse_skip(Some(&Value::Num(i64::MAX as f64))).is_err());
    assert!(parse_skip(Some(&Value::Num((i64::MAX as f64) + 1.0))).is_err());
}

#[test]
fn fwrite_fid_and_skip_read_typed_integer_storage_exactly() {
    let fid = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("fid tensor");
    assert_eq!(parse_fid(&Value::Tensor(fid)).unwrap(), 7);
    assert_eq!(parse_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
    assert!(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());

    let skip = Tensor::new_integer(IntegerStorage::U16(vec![9]), vec![1, 1]).expect("skip tensor");
    assert_eq!(parse_skip(Some(&Value::Tensor(skip))).unwrap(), 9);

    let too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("skip");
    assert!(parse_skip(Some(&Value::Tensor(too_large))).is_err());
}

#[test]
fn fwrite_typed_scalar_parameters_ignore_poisoned_f64_mirrors() {
    let classes = [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ];
    for storage in classes {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed scalar");
        let value = Value::Tensor(tensor);
        assert_eq!(parse_fid(&value).unwrap(), 7);
        assert_eq!(parse_skip(Some(&value)).unwrap(), 7);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_default_uint8_bytes() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_uint8");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![1.0, 2.0, 255.0], vec![3, 1]).unwrap();
    let eval =
        run_evaluate(&Value::Num(fid as f64), &Value::Tensor(tensor), &Vec::new()).expect("fwrite");
    assert_eq!(eval.count(), 3);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(bytes, vec![1u8, 2, 255]);
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_uint64_preserves_exact_typed_tensor_bytes() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_uint64_exact");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let values = [9_007_199_254_740_993, u64::MAX];
    let tensor = Tensor::new_integer(IntegerStorage::U64(values.to_vec()), vec![2, 1])
        .expect("typed uint64 tensor");

    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &Value::Tensor(tensor),
        &[Value::from("uint64")],
    )
    .expect("fwrite");
    assert_eq!(eval.count(), values.len());
    run_fclose(&[Value::Num(fid as f64)]).expect("fclose");

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(
        bytes,
        values
            .into_iter()
            .flat_map(u64::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    test_support::fs::remove_file(path).expect("remove file");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_int64_preserves_exact_typed_tensor_bytes() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_int64_exact");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let values = [i64::MIN, i64::MAX];
    let tensor = Tensor::new_integer(IntegerStorage::I64(values.to_vec()), vec![2, 1])
        .expect("typed int64 tensor");

    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &Value::Tensor(tensor),
        &[Value::from("int64")],
    )
    .expect("fwrite");
    assert_eq!(eval.count(), values.len());
    run_fclose(&[Value::Num(fid as f64)]).expect("fclose");

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(
        bytes,
        values
            .into_iter()
            .flat_map(i64::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    test_support::fs::remove_file(path).expect("remove file");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_uint64_narrowing_stays_in_integer_domain() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_uint64_to_uint32");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
        Value::from("ieee-be"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2, 1],
    )
    .expect("typed uint64 tensor");

    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &Value::Tensor(tensor),
        &[Value::from("uint32")],
    )
    .expect("fwrite");
    assert_eq!(eval.count(), 2);
    run_fclose(&[Value::Num(fid as f64)]).expect("fclose");

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(
        bytes,
        [u32::MAX.to_be_bytes(), u32::MAX.to_be_bytes()].concat()
    );
    test_support::fs::remove_file(path).expect("remove file");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fwrite_signed_integer_to_unsigned_target_saturates_without_f64_rounding() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fwrite_int64_to_uint16");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w+b"),
        Value::from("ieee-be"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let tensor = Tensor::new_integer(
        IntegerStorage::I64(vec![i64::MIN, -1, 65_535, i64::MAX]),
        vec![4, 1],
    )
    .expect("typed int64 tensor");

    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &Value::Tensor(tensor),
        &[Value::from("uint16")],
    )
    .expect("fwrite");
    assert_eq!(eval.count(), 4);
    run_fclose(&[Value::Num(fid as f64)]).expect("fclose");

    let bytes = test_support::fs::read(&path).expect("read");
    assert_eq!(
        bytes,
        [
            0_u16.to_be_bytes(),
            0_u16.to_be_bytes(),
            65_535_u16.to_be_bytes(),
            u16::MAX.to_be_bytes(),
        ]
        .concat()
    );
    test_support::fs::remove_file(path).expect("remove file");
}
