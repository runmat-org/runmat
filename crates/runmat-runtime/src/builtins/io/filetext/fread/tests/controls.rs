use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn double_skip_checked_against_platform_width_before_conversion() {
    let value = 4_294_967_296.0;
    #[cfg(target_pointer_width = "32")]
    assert_eq!(
        parse_skip_scalar(value),
        Err("skip value is too large".into())
    );
    #[cfg(target_pointer_width = "64")]
    assert_eq!(parse_skip_scalar(value), Ok(4_294_967_296_usize));
    assert!(parse_skip_scalar(i64::MAX as f64).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn precision_text_accepts_trailing_singletons_but_not_character_pages() {
    let row = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("uint8".chars().collect(), vec![1, 5, 1, 1])
            .unwrap(),
    );
    assert!(is_string_like(&row));
    assert!(parse_precision(Some(&row)).is_ok());
    let pages = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("uint8".chars().collect(), vec![1, 1, 5])
            .unwrap(),
    );
    assert!(!is_string_like(&pages));
    assert!(parse_precision(Some(&pages)).is_err());
}
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"data = fread(fid)"));
    assert!(labels.contains(&"data = fread(fid, size, precision, skip, machinefmt)"));
    assert!(labels.contains(&"data = fread(fid, precision, machinefmt)"));
    assert!(labels.contains(&"data = fread(fid, ..., \"like\", prototype)"));
}

#[test]
fn fread_integer_capabilities_and_control_roles_are_independently_gated() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 5);
    assert_eq!(INTEGER_CAPABILITIES[4].inputs[0].classes.len(), 8);
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let fid = preflight_control(&Value::Int(IntValue::I32(3)), ControlRole::FileId).unwrap_err();
    assert_eq!(
        fid.identifier(),
        Some("RunMat:compatibility:FreadIntegerIdExtension")
    );
    let size = preflight_control(&Value::Int(IntValue::U16(2)), ControlRole::Size).unwrap_err();
    assert_eq!(
        size.identifier(),
        Some("RunMat:compatibility:FreadIntegerSizeExtension")
    );
    let skip = preflight_control(&Value::Int(IntValue::U8(1)), ControlRole::Skip).unwrap_err();
    assert_eq!(
        skip.identifier(),
        Some("RunMat:compatibility:FreadIntegerSkipExtension")
    );
}

#[test]
fn fread_classifies_resident_size_and_skip_before_gathering() {
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
    let precision = Value::from("uint8");
    let args = [resident.clone(), precision, resident];
    let (size, _, skip, _, _) =
        classify_arguments(crate::call::FunctionArgumentView::from_slice(&args))
            .expect("classified controls");
    assert!(matches!(size, Some(Value::GpuTensor(_))));
    assert!(matches!(skip, Some(Value::GpuTensor(_))));
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = preflight_control(skip.expect("skip"), ControlRole::Skip).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FreadResidentControlExtension")
    );
}

#[test]
fn fread_like_gate_precedes_resident_fid_gather_and_effects() {
    let fid = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let args = [Value::from("like"), Value::Num(0.0)];
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run_evaluate(&fid, &args).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FreadLikeExtension")
    );
}

#[test]
fn fread_invalid_like_prototype_does_not_advance_file_position() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_invalid_like_position");
    test_support::fs::write(&path, [11_u8, 22_u8]).expect("write fixture");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("open");
    let fid = open.as_open().expect("open result").fid as i32;
    let unsupported =
        Value::Cell(runmat_value::CellArray::new(vec![Value::Num(0.0)], 1, 1).expect("cell"));
    let error = run_evaluate(
        &Value::Num(fid as f64),
        &[
            Value::Num(1.0),
            Value::from("uint8"),
            Value::from("like"),
            unsupported,
        ],
    )
    .unwrap_err();
    assert!(error
        .message()
        .contains("cell prototypes are not supported"));

    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &[Value::Num(1.0), Value::from("*uint8")],
    )
    .expect("read after rejected prototype");
    let Value::Tensor(tensor) = eval.data() else {
        panic!("expected tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U8(vec![11]))
    );
    run_fclose(&[Value::Num(fid as f64)]).expect("close");
    test_support::fs::remove_file(path).expect("remove fixture");
}

#[test]
fn fread_numeric_like_with_character_precision_does_not_advance_file_position() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_numeric_like_char_position");
    test_support::fs::write(&path, [31_u8, 32_u8]).expect("write fixture");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("open");
    let fid = open.as_open().expect("open result").fid as i32;
    let error = run_evaluate(
        &Value::Num(fid as f64),
        &[
            Value::Num(1.0),
            Value::from("*char"),
            Value::from("like"),
            Value::Num(0.0),
        ],
    )
    .unwrap_err();
    assert!(error
        .message()
        .contains("character output requires a character prototype"));
    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &[Value::Num(1.0), Value::from("*uint8")],
    )
    .expect("read after rejected matrix");
    let Value::Tensor(tensor) = eval.data() else {
        panic!("expected tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U8(vec![31]))
    );
    run_fclose(&[Value::Num(fid as f64)]).expect("close");
    test_support::fs::remove_file(path).expect("remove fixture");
}

#[test]
fn fread_resident_like_with_character_precision_does_not_advance_file_position() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _registry = registry_guard();
    let _guard = test_support::accel_test_lock();
    let owner: &'static FreadOwnerProvider = Box::leak(Box::new(FreadOwnerProvider::new(
        runmat_accelerate_api::ProviderPrecision::F64,
    )));
    unsafe { runmat_accelerate_api::register_provider(owner) };
    let prototype = owner
        .upload(&HostTensorView {
            data: &[0.0],
            shape: &[1, 1],
        })
        .expect("prototype upload");
    registry::reset_for_tests();
    let path = unique_path("fread_resident_like_char_position");
    test_support::fs::write(&path, [41_u8, 42_u8]).expect("write fixture");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("open");
    let fid = open.as_open().expect("open result").fid as i32;
    let error = run_evaluate(
        &Value::Num(fid as f64),
        &[
            Value::Num(1.0),
            Value::from("*char"),
            Value::from("like"),
            Value::GpuTensor(prototype),
        ],
    )
    .unwrap_err();
    assert!(error
        .message()
        .contains("character output requires a character prototype"));
    let eval = run_evaluate(
        &Value::Num(fid as f64),
        &[Value::Num(1.0), Value::from("*uint8")],
    )
    .expect("read after rejected resident matrix");
    let Value::Tensor(tensor) = eval.data() else {
        panic!("expected tensor");
    };
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U8(vec![41]))
    );
    run_fclose(&[Value::Num(fid as f64)]).expect("close");
    test_support::fs::remove_file(path).expect("remove fixture");
}

#[test]
fn fread_empty_count_returns_an_empty_column() {
    let mut cooperation =
        crate::builtins::io::filetext::cooperation::FileCooperation::new(None, BUILTIN_NAME);
    let (storage, rows, cols) = futures::executor::block_on(finalize_numeric(
        &SizeSpec::Count(0),
        0,
        NumericStorage::U8(Vec::new()),
        &mut cooperation,
    ))
    .expect("empty result");
    assert_eq!((rows, cols), (0, 1));
    assert_eq!(storage.len(), 0);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn fread_size_tensor_parser_preserves_exact_integer_storage() {
    let count = (1_u64 << 53) + 1;
    let count_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![count]), vec![1, 1]).expect("count");
    match parse_size(Some(&Value::Tensor(count_tensor))).expect("size") {
        SizeSpec::Count(value) => assert_eq!(value, usize::try_from(count).unwrap()),
        other => panic!("expected count size, got {other:?}"),
    }

    let matrix_tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![count, 3]), vec![1, 2]).expect("matrix");
    match parse_size(Some(&Value::Tensor(matrix_tensor))).expect("size") {
        SizeSpec::Matrix {
            rows,
            cols: Some(cols),
        } => {
            assert_eq!(rows, usize::try_from(count).unwrap());
            assert_eq!(cols, 3);
        }
        other => panic!("expected matrix size, got {other:?}"),
    }
}

#[test]
fn fread_size_tensor_parser_rejects_negative_integer_storage() {
    let count_tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("count");
    assert!(parse_size(Some(&Value::Tensor(count_tensor))).is_err());

    let matrix_tensor =
        Tensor::new_integer(IntegerStorage::I16(vec![2, -1]), vec![1, 2]).expect("matrix");
    assert!(parse_size(Some(&Value::Tensor(matrix_tensor))).is_err());
}

#[test]
fn fread_scalar_parser_reads_typed_integer_storage_exactly() {
    let scalar = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("scalar");
    assert_eq!(
        value_to_scalar(&Value::Tensor(scalar), "scalar").expect("scalar"),
        7.0
    );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn fread_scalar_size_and_skip_parse_integer_values_exactly() {
    let exact = (1_u64 << 53) + 1;
    match parse_size(Some(&Value::Int(IntValue::U64(exact)))).expect("size") {
        SizeSpec::Count(value) => assert_eq!(value, exact as usize),
        other => panic!("expected count size, got {other:?}"),
    }

    assert_eq!(
        parse_skip(Some(&Value::Int(IntValue::U64(exact)))).unwrap(),
        exact as usize
    );
    assert!(parse_skip(Some(&Value::Int(IntValue::U64(u64::MAX)))).is_err());
    assert!(parse_size(Some(&Value::Int(IntValue::I8(-1)))).is_err());

    assert!(parse_size(Some(&Value::Num(usize::MAX as f64))).is_err());
    assert!(parse_size(Some(&Value::Num((usize::MAX as f64) + 1.0))).is_err());
    assert!(parse_skip(Some(&Value::Num(i64::MAX as f64))).is_err());
    assert!(parse_skip(Some(&Value::Num((i64::MAX as f64) + 1.0))).is_err());
}

#[test]
fn fread_fid_and_skip_read_typed_integer_storage_exactly() {
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
fn fread_typed_scalar_parameters_cover_every_integer_class() {
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
        assert!(matches!(
            parse_size(Some(&value)).unwrap(),
            SizeSpec::Count(7)
        ));
        assert_eq!(parse_skip(Some(&value)).unwrap(), 7);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fread_default_reads_uint8_and_returns_double() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fread_default_double");
    let mut file = File::create(&path).expect("create");
    file.write_all(&[7_u8]).expect("write");
    drop(file);

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let eval = run_evaluate(&Value::Num(fid as f64), &Vec::new()).expect("fread");
    assert_eq!(eval.count(), 1);
    match eval.data() {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[1, 1]);
            assert_eq!(double_values(t), &[7.0]);
        }
        other => panic!("unexpected result {other:?}"),
    }

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}
