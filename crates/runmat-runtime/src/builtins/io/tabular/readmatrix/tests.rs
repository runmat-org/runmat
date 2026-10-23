use super::*;
use futures::executor::block_on;
use runmat_filesystem::File;
use runmat_time::unix_timestamp_ms;
use std::fs;

use crate::builtins::common::test_support;

#[path = "tests/owner.rs"]
mod owner;

#[path = "tests/output.rs"]
mod output;

#[test]
fn resident_like_logical_upload_is_owned_and_preserves_nd_empty_shapes() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1],
            })
            .unwrap();
        for (data, shape) in [
            (vec![0.0, -2.0, 3.0, 0.0], vec![2, 1, 2]),
            (vec![], vec![0, 3, 2]),
        ] {
            let expected = data
                .iter()
                .map(|value| if *value == 0.0 { 0.0 } else { 1.0 })
                .collect::<Vec<_>>();
            let Value::GpuTensor(output) =
                tensor_to_gpu(Tensor::new(data, shape.clone()).unwrap(), &prototype).unwrap()
            else {
                panic!("resident Like result");
            };
            assert!(output.view().is_logical());
            assert!(!output.shares_allocation_with(&prototype));
            assert_eq!(output.shape(), shape);
            assert_eq!(
                futures::executor::block_on(provider.download(&output))
                    .unwrap()
                    .data,
                expected
            );
        }
        assert!(futures::executor::block_on(provider.download(&prototype)).is_ok());
    });
}

#[test]
fn logical_like_rejects_wrong_shape_without_retiring_prototype() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1],
            })
            .expect("prototype");
        let invalid = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1],
                shape: &[2, 1],
            })
            .expect("wrong-shaped provider result");
        let error = super::output::checked_like_result(invalid, &prototype, &[1, 2])
            .expect_err("logical Like must validate the returned shape");
        assert_eq!(error.identifier(), READMATRIX_ERROR_OUTPUT_TYPE.identifier);
        assert!(block_on(provider.download(&prototype)).is_ok());
    });
}

#[test]
fn logical_like_retires_wrong_owner_result_and_preserves_caller_alias() {
    use runmat_accelerate_api::test_support::TestAllocationOwner;

    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1],
            })
            .expect("prototype");
        let foreign = TestAllocationOwner::new()
            .allocate_logical(vec![0], vec![1, 1])
            .expect("foreign logical allocation");
        let probe = foreign.probe;
        let error = super::output::checked_like_result(foreign.handle, &prototype, &[1, 1])
            .expect_err("a foreign provider result cannot be published");
        assert_eq!(error.identifier(), READMATRIX_ERROR_OUTPUT_TYPE.identifier);
        assert_eq!(probe.retirements(), 1, "rejected result is retired");

        let error = super::output::checked_like_result(prototype.clone(), &prototype, &[1, 1])
            .expect_err("Like must not return the caller's own allocation");
        assert_eq!(error.identifier(), READMATRIX_ERROR_OUTPUT_TYPE.identifier);
        assert!(block_on(provider.download(&prototype)).is_ok());
    });
}

#[test]
fn logical_like_rejects_numeric_class_from_same_provider() {
    test_support::with_test_provider(|provider| {
        let prototype = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[1],
                shape: &[1, 1],
            })
            .expect("logical prototype");
        let numeric = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0], vec![1, 1]).expect("numeric source"),
        )
        .expect("numeric provider result");
        let error = super::output::checked_like_result(numeric, &prototype, &[1, 1])
            .expect_err("numeric storage cannot satisfy logical Like");
        assert_eq!(error.identifier(), READMATRIX_ERROR_OUTPUT_TYPE.identifier);
        assert!(block_on(provider.download(&prototype)).is_ok());
    });
}
use runmat_value::{CharArray, IntValue, LogicalArray, StringArray, Tensor};

fn unique_path(prefix: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let mut path = std::env::temp_dir();
    path.push(format!("runmat_{prefix}_{}_{}", std::process::id(), millis));
    path
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[test]
fn readmatrix_applies_typed_per_column_missing_and_numeric_rules() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_variable_options");
    fs::write(&path, "N/A,1.234\n2,2.500\n").expect("write numeric fixture");
    let mut base = runmat_value::UnboundedStructValueBuilder::new();
    base.insert("FileType", Value::from("text"));
    base.insert("NumVariables", Value::Num(2.0));
    base.insert("Delimiter", Value::from(","));
    base.insert(
        "VariableNames",
        Value::StringArray(StringArray::new(vec!["A".into(), "B".into()], vec![1, 2]).unwrap()),
    );
    base.insert(
        "VariableTypes",
        Value::StringArray(
            StringArray::new(vec!["double".into(), "double".into()], vec![1, 2]).unwrap(),
        ),
    );
    let base = base.finish();
    let mut options = ImportOptions::from_legacy(base).expect("typed import options");
    options
        .set_options(
            Some(&Value::from("A")),
            &[
                ("TreatAsMissing".into(), Value::from("N/A")),
                ("FillValue".into(), Value::Num(0.0)),
            ],
        )
        .expect("first column controls");
    options
        .set_options(
            Some(&Value::from("B")),
            &[
                ("DecimalSeparator".into(), Value::from(",")),
                ("ThousandsSeparator".into(), Value::from(".")),
            ],
        )
        .expect("second column controls");
    let output = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![options
            .to_value(None)
            .expect("encode import options")]),
    ))
    .expect("readmatrix with per-variable options");
    let Value::Tensor(output) = output else {
        panic!("expected numeric matrix")
    };
    assert_eq!(output.materialize_f64(), vec![0.0, 2.0, 1234.0, 2500.0]);
    let _ = fs::remove_file(path);
}

#[test]
fn readmatrix_like_preserves_every_exact_integer_class() {
    let prototypes = [
        IntegerStorage::I8(vec![0]),
        IntegerStorage::I16(vec![0]),
        IntegerStorage::I32(vec![0]),
        IntegerStorage::I64(vec![0]),
        IntegerStorage::U8(vec![0]),
        IntegerStorage::U16(vec![0]),
        IntegerStorage::U32(vec![0]),
        IntegerStorage::U64(vec![0]),
    ];

    for storage in prototypes {
        let expected = storage
            .from_same_class_values(
                [1.0, 2.5, -3.0]
                    .into_iter()
                    .map(|value| storage.cast_f64_assignment(value))
                    .collect(),
            )
            .expect("expected storage");
        let source = Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap();
        let prototype = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
        let output = finalize_like(source, &prototype).expect("integer like output");
        let Value::Tensor(output) = output else {
            panic!("expected tensor output");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }

    let source = Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap();
    let prototype = Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap());
    let output = finalize_like(source, &prototype).expect("single like output");
    let Value::Tensor(output) = output else {
        panic!("expected single tensor output");
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(values(&output), vec![1.0, 2.5, -3.0]);
}

#[test]
fn readmatrix_output_type_supports_every_numeric_class() {
    let cases = [
        ("int8", IntegerStorage::I8(vec![1, 3, -3])),
        ("int16", IntegerStorage::I16(vec![1, 3, -3])),
        ("int32", IntegerStorage::I32(vec![1, 3, -3])),
        ("int64", IntegerStorage::I64(vec![1, 3, -3])),
        ("uint8", IntegerStorage::U8(vec![1, 3, 0])),
        ("uint16", IntegerStorage::U16(vec![1, 3, 0])),
        ("uint32", IntegerStorage::U32(vec![1, 3, 0])),
        ("uint64", IntegerStorage::U64(vec![1, 3, 0])),
    ];
    for (class_name, expected) in cases {
        let mut options = ReadMatrixOptions::default();
        options
            .set_output_type(class_name)
            .expect("numeric output type");
        let source = Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap();
        let Value::Tensor(output) = finalize_output(source, &options).expect("numeric output")
        else {
            panic!("expected numeric tensor output");
        };
        assert_eq!(output.integer_storage(), Some(&expected));
    }

    let mut single_options = ReadMatrixOptions::default();
    single_options
        .set_output_type("single")
        .expect("single output type");
    let source = Tensor::new(vec![1.0, 2.5, -3.0], vec![3, 1]).unwrap();
    let Value::Tensor(single) = finalize_output(source, &single_options).expect("single output")
    else {
        panic!("expected single tensor output");
    };
    assert_eq!(single.numeric_dtype(), NumericDType::F32);
    assert_eq!(values(&single), vec![1.0, 2.5, -3.0]);
}

#[test]
fn readmatrix_output_type_parses_wide_integer_text_without_binary64() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_wide_integer");
    fs::write(
        &path,
        "9007199254740993,18446744073709551615\n9223372036854775808,7\n",
    )
    .expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("OutputType"),
            Value::from("uint64"),
        ]),
    ))
    .expect("readmatrix");
    let Value::Tensor(result) = result else {
        panic!("expected tensor");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            1_u64 << 63,
            u64::MAX,
            7,
        ]))
    );
    let _ = fs::remove_file(path);
}

#[test]
fn readmatrix_documented_integer_output_remains_enabled_in_matlab_mode() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_integer_matlab_mode");
    fs::write(&path, "9007199254740993\n").expect("write sample file");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("OutputType"),
            Value::from("uint64"),
        ]),
    ))
    .expect("documented integer output");
    let Value::Tensor(result) = result else {
        panic!("expected integer tensor");
    };
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993]))
    );
    let _ = fs::remove_file(path);
}

#[test]
fn readmatrix_compatibility_gates_extensions_before_file_access() {
    let missing = Value::from("runmat_missing_readmatrix_input.csv");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);

    let like_error = block_on(readmatrix_builtin(
        None,
        missing.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Like"),
            Value::Int(IntValue::U8(0)),
        ]),
    ))
    .expect_err("Like must be gated");
    assert_eq!(
        like_error.identifier(),
        READMATRIX_LIKE_EXTENSION.error_identifier
    );

    let control_error = block_on(readmatrix_builtin(
        None,
        missing,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("NumHeaderLines"),
            Value::Int(IntValue::U8(1)),
        ]),
    ))
    .expect_err("typed control must be gated");
    assert_eq!(
        control_error.identifier(),
        READMATRIX_TYPED_INTEGER_CONTROL_EXTENSION.error_identifier
    );
}

#[test]
fn readmatrix_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = READMATRIX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(
        labels.contains(&"M = readmatrix(filename)"),
        "missing core readmatrix signature label"
    );
    assert!(
        labels.contains(&"M = readmatrix(filename, options)"),
        "missing options-struct readmatrix signature label"
    );
    assert!(
        labels.contains(&"M = readmatrix(filename, name, optionValue)"),
        "missing name-value readmatrix signature label"
    );
    assert!(
        labels.contains(&"M = readmatrix(filename, options, name, optionValue)"),
        "missing options+name-value readmatrix signature label"
    );
    assert!(
        labels.contains(&"M = readmatrix(filename, nameValuePairs...)"),
        "missing variadic name-value readmatrix signature label"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_reads_csv_data() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_csv");
    fs::write(&path, "1,2,3\n4,5,6\n").expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(values(&t), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn treat_as_missing_preserves_exact_uint64_text() {
    let tokens = block_on(parse_treat_as_missing(&Value::Int(IntValue::U64(u64::MAX))))
        .expect("TreatAsMissing token");
    assert_eq!(tokens, vec!["18446744073709551615"]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn treat_as_missing_reads_typed_integer_tensor_storage_exactly() {
    let wide = (1_u64 << 53) + 1;
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("token");

    let tokens =
        block_on(parse_treat_as_missing(&Value::Tensor(tensor))).expect("TreatAsMissing token");
    assert_eq!(tokens, vec!["9007199254740993"]);
}

#[test]
fn empty_value_reads_typed_integer_tensor_storage_exactly() {
    assert!(matches!(
        value_to_numeric_scalar(&Value::Num(f64::NAN), "EmptyValue"),
        Ok(NumericScalar::F64(value)) if value.is_nan()
    ));
    let wide = 9_007_199_254_740_993_u64;
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![wide]), vec![1, 1]).expect("empty value");
    let value = value_to_numeric_scalar(&Value::Tensor(tensor), "EmptyValue").expect("EmptyValue");
    assert_eq!(value, NumericScalar::U64(wide));

    let mut options = ReadMatrixOptions {
        empty_value: Some(value),
        ..ReadMatrixOptions::default()
    };
    assert_eq!(
        parse_integer_numeric_token(
            "",
            &options,
            NumericDType::U64,
            1,
            options.column_from_source(0)
        )
        .unwrap(),
        NumericScalar::U64(wide)
    );
    options.add_missing_token("NA");
    assert_eq!(
        parse_integer_numeric_token(
            "NA",
            &options,
            NumericDType::U64,
            1,
            options.column_from_source(0)
        )
        .unwrap(),
        NumericScalar::U64(0)
    );
    assert_eq!(
        parse_integer_numeric_token(
            "999999999999999999999999999999999999999999999999",
            &options,
            NumericDType::U64,
            1,
            options.column_from_source(0),
        )
        .unwrap(),
        NumericScalar::U64(u64::MAX)
    );
    assert_eq!(
        parse_integer_numeric_token(
            "-999999999999999999999999999999999999999999999999",
            &options,
            NumericDType::I64,
            1,
            options.column_from_source(0),
        )
        .unwrap(),
        NumericScalar::I64(i64::MIN)
    );
}

#[test]
fn numeric_range_reads_typed_integer_tensor_storage_exactly() {
    let tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![2, 3, 4, 5]), vec![1, 4]).expect("range");

    let range = parse_range_numeric(&Value::Tensor(tensor)).expect("range");
    assert_eq!(range.start_row, 1);
    assert_eq!(range.start_col, 2);
    assert_eq!(range.end_row, Some(3));
    assert_eq!(range.end_col, Some(4));
}

#[test]
fn option_usize_preserves_representable_uint64_values() {
    assert_eq!(
        value_to_usize(&Value::Int(IntValue::U64(u64::MAX)), "option").ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(value_to_usize(&Value::Int(IntValue::I64(-1)), "option").is_err());

    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1]).expect("option");
    assert_eq!(value_to_usize(&Value::Tensor(tensor), "option").unwrap(), 9);

    let negative = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("option");
    assert!(value_to_usize(&Value::Tensor(negative), "option").is_err());
    assert!(value_to_usize(&Value::Num(1.0e300), "option").is_err());
}

#[test]
fn option_usize_reads_every_integer_class_exactly() {
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
        let value = Tensor::new_integer(storage, vec![1, 1]).expect("typed option");
        assert_eq!(value_to_usize(&Value::Tensor(value), "option").unwrap(), 7);
    }

    let too_wide =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("wide");
    assert_eq!(
        value_to_usize(&Value::Tensor(too_wide), "option").ok(),
        usize::try_from(u64::MAX).ok()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_skips_header_lines() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("readmatrix_header");
    fs::write(&path, "time,value\n0,10\n1,12\n").expect("write sample file");
    let args = vec![Value::from("NumHeaderLines"), Value::Int(IntValue::I32(1))];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![0.0, 1.0, 10.0, 12.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_respects_delimiter_option() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_tab");
    fs::write(&path, "1\t2\t3\n4\t5\t6\n").expect("write sample file");
    let args = vec![Value::from("Delimiter"), Value::from("tab")];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(values(&t), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_respects_range_string() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_range_string");
    fs::write(&path, "11,12,13\n21,22,23\n31,32,33\n").expect("write sample file");
    let args = vec![Value::from("Range"), Value::from("B2:C3")];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![22.0, 32.0, 23.0, 33.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_respects_range_numeric_vector() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_range_numeric");
    fs::write(&path, "11,12,13\n21,22,23\n31,32,33\n").expect("write sample file");
    let range = Tensor::new(vec![2.0, 2.0, 3.0, 3.0], vec![1, 4]).expect("range tensor");
    let args = vec![Value::from("Range"), Value::Tensor(range)];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![22.0, 32.0, 23.0, 33.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_treats_custom_missing_tokens() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_missing");
    fs::write(&path, "1,NA,3\nNA,5,missing\n").expect("write file");
    let strings = StringArray::new(vec!["NA".to_string(), "missing".to_string()], vec![1, 2])
        .expect("string array");
    let args = vec![Value::from("TreatAsMissing"), Value::StringArray(strings)];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 3]);
            let values = values(&t);
            assert!(values[1].is_nan()); // column 1, row 2
            assert!(values[2].is_nan()); // column 2, row 1
            assert!(values[5].is_nan()); // column 3, row 2
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_uses_decimal_and_thousands_separators() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_decimal");
    fs::write(&path, "1.234,56;7.890,12\n").expect("write sample file");
    let args = vec![
        Value::from("Delimiter"),
        Value::from(";"),
        Value::from("DecimalSeparator"),
        Value::from(","),
        Value::from("ThousandsSeparator"),
        Value::from("."),
    ];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 2]);
            let values = values(&t);
            assert!((values[0] - 1234.56).abs() < 1e-9);
            assert!((values[1] - 7890.12).abs() < 1e-9);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_applies_empty_value() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_empty_value");
    fs::write(&path, "1,,3\n4,,6\n").expect("write sample file");
    let args = vec![Value::from("EmptyValue"), Value::Num(0.0)];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(values(&t), vec![1.0, 4.0, 0.0, 0.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_accepts_struct_options() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("readmatrix_struct_opts");
    fs::write(&path, "header1,header2\n9,10\n11,12\n").expect("write sample file");
    let mut options_struct = runmat_value::UnboundedStructValueBuilder::new();
    options_struct
        .fields
        .insert("Delimiter".to_string(), Value::from(","));
    options_struct
        .fields
        .insert("NumHeaderLines".to_string(), Value::Int(IntValue::I32(1)));
    let args = vec![Value::Struct(options_struct.finish())];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![9.0, 11.0, 10.0, 12.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_errors_on_non_numeric_field() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_error");
    fs::write(&path, "1,abc,3\n").expect("write sample file");
    let err = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("readmatrix should fail");
    let message = err.message().to_string();
    assert!(
        message.contains("unable to parse numeric value"),
        "unexpected error message: {message}"
    );
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_returns_empty_on_no_data() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_empty");
    File::create(&path).expect("create file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![0, 0]);
            assert!(t.is_empty());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_output_type_logical() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_output_logical");
    fs::write(&path, "0,1,-3\nNaN,0,5\n").expect("write sample file");
    let args = vec![Value::from("OutputType"), Value::from("logical")];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![2, 3]);
            assert_eq!(arr.data, vec![0, 1, 1, 0, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_like_logical_proto() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("readmatrix_like_logical");
    fs::write(&path, "1,0\n0,5\n").expect("write sample file");
    let proto = LogicalArray::new(vec![1], vec![1]).expect("logical prototype");
    let args = vec![Value::from("Like"), Value::LogicalArray(proto)];
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect("readmatrix");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![2, 2]);
            assert_eq!(arr.data, vec![1, 0, 0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_like_gpu_proto() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let _fs_lock = runmat_filesystem::provider_override_lock();
        let path = unique_path("readmatrix_like_gpu");
        fs::write(&path, "1,2\n3,4\n").expect("write sample file");
        let proto_tensor = Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("tensor");
        let handle = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload prototype");
        let args = vec![Value::from("Like"), Value::GpuTensor(handle.clone())];
        let result = block_on(readmatrix_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("readmatrix");
        let Value::GpuTensor(output) = &result else {
            panic!("expected GPU tensor result, got {result:?}");
        };
        assert!(
            gpu_helpers::exact_provider_for_handle(output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(output),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        assert_eq!(
            runmat_accelerate_api::handle_provenance(output),
            Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        );
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.shape(), vec![2, 2]);
        assert_eq!(values(&gathered), vec![1.0, 3.0, 2.0, 4.0]);
        let _ = fs::remove_file(&path);
    });
}

#[test]
fn readmatrix_integer_gpu_like_preserves_wide_storage_and_exact_owner() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let _fs_lock = runmat_filesystem::provider_override_lock();
        let path = unique_path("readmatrix_integer_gpu_like");
        fs::write(&path, "9007199254740993,18446744073709551615\n").expect("write sample file");
        let prototype = Tensor::new_integer(IntegerStorage::U64(vec![0]), vec![1, 1])
            .expect("integer prototype");
        let prototype =
            gpu_helpers::upload_tensor(provider, &prototype).expect("upload integer prototype");
        let result = block_on(readmatrix_builtin(
            None,
            Value::from(path.to_string_lossy().to_string()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("Like"),
                Value::GpuTensor(prototype),
            ]),
        ))
        .expect("integer Like readmatrix");
        let Value::GpuTensor(output) = &result else {
            panic!("expected resident integer output");
        };
        assert!(
            gpu_helpers::exact_provider_for_handle(output).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let gathered = test_support::gather(result).expect("gather integer output");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
        );
        let _ = fs::remove_file(path);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_accepts_character_vector_path() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_char_path");
    fs::write(&path, "1 2 3\n").expect("write sample file");
    let text = path.to_string_lossy().to_string();
    let chars: Vec<char> = text.chars().collect();
    let len = text.encode_utf16().count();
    let char_array = CharArray::from_unicode_scalars(chars, 1, len).expect("char array");
    let result = block_on(readmatrix_builtin(
        None,
        Value::CharArray(char_array),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 3]);
            assert_eq!(values(&t), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_handles_quoted_fields() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_quotes");
    fs::write(&path, "\"1\",\"2\",\"3\"\n").expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 3]);
            assert_eq!(values(&t), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_preserves_negative_infinity() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_infinity");
    fs::write(&path, "-Inf,Inf,NaN\n").expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![1, 3]);
            let values = values(&t);
            assert!(values[0].is_infinite() && values[0].is_sign_negative());
            assert!(values[1].is_infinite() && values[1].is_sign_positive());
            assert!(values[2].is_nan());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_accepts_shift_jis_encoding_label() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_shift_jis");
    fs::write(&path, b"1,2\n3,4\n").expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Encoding"),
            Value::from("shift_jis"),
        ]),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_auto_detects_utf16_bom() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_utf16_auto");
    let mut bytes = vec![0xFF, 0xFE];
    for unit in "1,2\n3,4\n".encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    fs::write(&path, bytes).expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("Encoding"),
            Value::from("auto"),
        ]),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(values(&t), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn readmatrix_supports_whitespace_delimiter() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = unique_path("readmatrix_whitespace");
    fs::write(&path, "1 2 3\n4 5 6\n").expect("write sample file");
    let result = block_on(readmatrix_builtin(
        None,
        Value::from(path.to_string_lossy().to_string()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("readmatrix");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(values(&t), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
    let _ = fs::remove_file(&path);
}
