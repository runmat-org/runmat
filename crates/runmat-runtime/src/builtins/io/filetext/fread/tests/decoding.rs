use super::*;
#[test]
fn fread_precision_parser_supports_every_numeric_output_class() {
    let cases = [
        ("uint8=>double", NumericDType::F64),
        ("uint8=>single", NumericDType::F32),
        ("uint8=>int8", NumericDType::I8),
        ("uint8=>int16", NumericDType::I16),
        ("uint8=>int32", NumericDType::I32),
        ("uint8=>int64", NumericDType::I64),
        ("uint8=>uint8", NumericDType::U8),
        ("uint8=>uint16", NumericDType::U16),
        ("uint8=>uint32", NumericDType::U32),
        ("uint8=>uint64", NumericDType::U64),
    ];
    for (precision, expected) in cases {
        let parsed = parse_precision_string(precision).expect("precision");
        assert!(matches!(parsed.output, OutputKind::Numeric(dtype) if dtype == expected));
    }
    assert!(matches!(
        parse_precision_string("*single").unwrap().output,
        OutputKind::Numeric(NumericDType::F32)
    ));
    assert!(matches!(
        parse_precision_string("*uint64").unwrap().output,
        OutputKind::Numeric(NumericDType::U64)
    ));
    let repeated = parse_precision_string("2*uint16=>uint16").unwrap();
    assert_eq!(repeated.repeat, 2);
    assert!(matches!(
        repeated.output,
        OutputKind::Numeric(NumericDType::U16)
    ));
    assert_eq!(parse_precision_string("real*4").unwrap().repeat, 1);
}

#[test]
fn fread_star_uint64_preserves_values_beyond_binary64_exact_range() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_exact_uint64");
    let expected = [9_007_199_254_740_993_u64, u64::MAX];
    let mut file = File::create(&path).expect("create");
    for value in expected {
        file.write_all(&value.to_le_bytes()).expect("write");
    }
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*uint64")]).expect("fread");
    assert_eq!(eval.count(), 2);
    let Value::Tensor(output) = eval.data() else {
        panic!("expected tensor output");
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(expected.to_vec()))
    );

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn fread_star_single_preserves_native_single_storage() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_native_single");
    let expected = [1.25_f32, -3.5_f32];
    let mut file = File::create(&path).expect("create");
    for value in expected {
        file.write_all(&value.to_le_bytes()).expect("write");
    }
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*single")]).expect("fread");
    let Value::Tensor(output) = eval.data() else {
        panic!("expected tensor output");
    };
    assert_eq!(
        output.clone().into_numeric_storage().expect("storage"),
        NumericStorage::F32(expected.to_vec())
    );

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn fread_logical_conversion_uses_typed_integer_storage() {
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        vec![3, 1],
    )
    .expect("integer tensor");

    let Value::LogicalArray(logical) =
        convert_to_logical_value(Value::Tensor(tensor)).expect("logical conversion")
    else {
        panic!("expected logical array");
    };
    assert_eq!(logical.shape, vec![3, 1]);
    assert_eq!(logical.data, vec![0, 1, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_uint8_vector_with_count() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_uint8_vector");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![Value::Num(4.0), Value::from("uint8")];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    assert_eq!(eval.count(), 4);
    match eval.data() {
        Value::Tensor(t) => {
            assert_eq!(double_values(t), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(t.shape(), &[4, 1]);
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_uint8_matrix_with_padding() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_uint8_matrix");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[1u8, 2, 3, 4, 5]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let size_tensor = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
    let args = vec![Value::Tensor(size_tensor), Value::from("uint8")];
    let eval = run_evaluate(&Value::Num(fid as f64), &args).expect("fread");
    assert_eq!(eval.count(), 5);
    match eval.data() {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(double_values(t), &[1.0, 2.0, 3.0, 4.0, 5.0, 0.0]);
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_char_output() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_char_output");
    let mut file = File::create(&path).expect("create");
    file.write_all(b"abc").expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let eval = run_evaluate(&Value::Num(fid as f64), &[Value::from("*char")]).expect("fread");
    assert_eq!(eval.count(), 3);
    match eval.data() {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 3);
            assert_eq!(ca.cols, 1);
            let collected: String = ca.data.iter().collect();
            assert_eq!(collected, "abc");
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_like_logical_output() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_like_logical_output");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[0u8, 3, 0, 4]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let prototype = LogicalArray::zeros(vec![2, 1]);
    let args = vec![
        Value::Num(2.0),
        Value::from("uint8"),
        Value::from("like"),
        Value::LogicalArray(prototype),
    ];
    {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = run_evaluate(&Value::Num(fid as f64), &args)
            .expect_err("MATLAB mode rejects fread like");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FreadLikeExtension")
        );
    }
    let eval = {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        run_evaluate(&Value::Num(fid as f64), &args).expect("RunMat mode accepts fread like")
    };
    assert_eq!(eval.count(), 2);
    match eval.data() {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 1]);
            assert_eq!(array.data, vec![0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_like_requires_prototype() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_like_requires_prototype");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[1u8, 2, 3, 4]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![Value::from("like")];
    let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
    assert!(err.contains("expected prototype after 'like'"));

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_like_char_requires_precision() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_like_char_requires_precision");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[65u8]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![
        Value::Num(1.0),
        Value::from("uint8"),
        Value::from("like"),
        Value::CharArray(CharArray::new_row("A")),
    ];
    let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &args).unwrap_err());
    assert!(err.contains("character prototypes require"));

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}
