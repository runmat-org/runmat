use super::super::entrypoints::{anymissing_builtin, ismissing_builtin, missing_builtin};
use super::*;
use runmat_value::ComplexTensor;

#[test]
fn typed_missing_and_literal_marker_remain_distinct_through_missing_operations() {
    use runmat_value::StringElement;
    let values = vec![
        StringElement::Missing,
        StringElement::text("<missing>"),
        StringElement::text("<MISSING>"),
        StringElement::text(""),
        StringElement::text("present"),
    ];
    let input = StringArray::from_elements_unbounded(values.clone(), vec![5, 1]).unwrap();
    let Value::LogicalArray(mask) = ismissing_value(&Value::StringArray(input.clone())).unwrap()
    else {
        panic!("expected missing mask");
    };
    assert_eq!(mask.data, vec![1, 0, 0, 0, 0]);
    for value in &values {
        assert_eq!(
            ismissing_value(&Value::String(value.clone())).unwrap(),
            Value::Bool(value.is_missing())
        );
    }
    let indicators = super::super::standardize::indicator_set(&Value::from("present")).unwrap();
    let Value::StringArray(output) = super::super::standardize::standardize_missing_value(
        None,
        Value::StringArray(input),
        &indicators,
    )
    .unwrap() else {
        panic!("expected standardized strings");
    };
    assert!(output.data()[0].is_missing());
    assert_eq!(&output.data()[1..4], &values[1..4]);
    assert!(output.data()[4].is_missing());
}

#[test]
fn ismissing_preserves_typed_struct_array_shape_and_schema() {
    let element = |value| {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("payload", value);
        structure.finish()
    };
    let array = StructArray::new(
        vec![
            element(Value::StringArray(
                StringArray::from_elements_unbounded(vec![MISSING_TEXT.into()], vec![1, 1])
                    .unwrap(),
            )),
            element(Value::String("present".into())),
        ],
        vec![2, 1],
    )
    .unwrap();
    let Value::StructArray(output) =
        ismissing_value(&Value::StructArray(array)).expect("ismissing structure array")
    else {
        panic!("expected typed structure array");
    };
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(output.field_names().collect::<Vec<_>>(), ["payload"]);
    let values = output.field_values("payload").unwrap();
    assert!(matches!(
        &values[0],
        Value::LogicalArray(mask) if mask.shape == [1, 1] && mask.data == vec![1]
    ));
    assert_eq!(values[1], Value::Bool(false));
}

#[test]
fn missing_constructs_scalar_and_arrays() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let scalar = block_on(missing_builtin(
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap();
    assert!(matches!(scalar, Value::StringArray(sa) if sa.data() == vec![MISSING_TEXT]));
    let shaped = block_on(missing_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
        ]),
    ))
    .unwrap();
    assert!(
        matches!(shaped, Value::StringArray(sa) if sa.shape == vec![2, 3] && sa.data().len() == 6)
    );
}

#[test]
fn missing_rejects_dangling_like_control() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(missing_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::from("like"),
        ]),
    ))
    .expect_err("a like keyword needs a prototype");
    assert_eq!(error.identifier(), Some("RunMat:missing:InvalidArgument"));
}

#[test]
fn missing_runmat_shape_extension_reads_every_integer_class_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let cases = [
        IntValue::I8(2),
        IntValue::I16(2),
        IntValue::I32(2),
        IntValue::I64(2),
        IntValue::U8(2),
        IntValue::U16(2),
        IntValue::U32(2),
        IntValue::U64(2),
    ];
    for size in cases {
        let result = block_on(missing_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(size)]),
        ))
        .expect("RunMat shaped missing");
        assert!(
            matches!(result, Value::StringArray(array) if array.shape == vec![2, 2] && array.data().len() == 4)
        );
    }
}

#[test]
fn missing_shaped_extension_gates_before_provider_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(gpu_helpers::exact_provider_for_handle(&allocation.handle).is_none());
    let value = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(missing_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![value]),
    ))
    .expect_err("MATLAB-compatible mode must reject shaped missing");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:MissingShapedArrayExtension")
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation.handle.release().unwrap();
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn missing_preserves_typed_integer_size_vectors_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let large = 9_007_199_254_740_993_u64;
    let dims = Tensor::new_integer(IntegerStorage::U64(vec![large, 0]), vec![1, 2]).unwrap();

    let result = block_on(missing_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(dims)]),
    ))
    .unwrap();

    match result {
        Value::StringArray(array) => {
            assert_eq!(array.shape, vec![large as usize, 0]);
            assert!(array.data().is_empty());
        }
        other => panic!("expected string array, got {other:?}"),
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn missing_parses_typed_integer_scalar_tensors_exactly() {
    let scalar = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]).unwrap(),
    );

    assert_eq!(
        scalar_usize(&scalar, "missing size").unwrap(),
        9_007_199_254_740_993
    );
}

#[test]
#[cfg(target_pointer_width = "32")]
fn missing_rejects_typed_integer_scalar_tensors_outside_platform_range() {
    let scalar = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1]).unwrap(),
    );

    let err = scalar_usize(&scalar, "missing size").expect_err("dimension must fit usize");
    assert!(err.message.contains("platform integer"), "{}", err.message);
}

#[test]
fn missing_numeric_scalar_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1])
        .expect("typed numeric scalar");

    assert_eq!(
        numeric_scalar(&Value::Tensor(tensor), "fillmissing constant").unwrap(),
        2026.0
    );
}

#[test]
fn missing_rejects_negative_typed_integer_sizes() {
    let scalar = Tensor::new_integer(IntegerStorage::I8(vec![-1]), vec![1, 1]).unwrap();
    assert!(scalar_usize(&Value::Tensor(scalar), "missing size").is_err());

    let dims = Tensor::new_integer(IntegerStorage::I64(vec![2, -1]), vec![1, 2]).unwrap();
    assert!(tensor_shape_as_size(&dims).is_err());
}

#[test]
fn missing_rejects_unrepresentable_double_sizes_before_casting() {
    let err = scalar_usize(
        &Value::Num(first_unrepresentable_usize_double()),
        "missing size",
    )
    .unwrap_err();
    assert!(err.message.contains("integer too large"), "{}", err.message);

    let dims = Tensor::new(vec![first_unrepresentable_usize_double(), 0.0], vec![1, 2]).unwrap();
    let err = tensor_shape_as_size(&dims).unwrap_err();
    assert!(err.message.contains("platform limits"), "{}", err.message);
}

#[test]
fn ismissing_detects_numeric_and_string_values() {
    let result = block_on(ismissing_builtin(tensor(
        vec![1.0, f64::NAN, 3.0],
        vec![1, 3],
    )))
    .unwrap();
    assert!(matches!(result, Value::LogicalArray(mask) if mask.data == vec![0, 1, 0]));

    let strings =
        StringArray::from_elements_unbounded(vec!["a".into(), MISSING_TEXT.into()], vec![1, 2])
            .unwrap();
    let result = block_on(ismissing_builtin(Value::StringArray(strings))).unwrap();
    assert!(matches!(result, Value::LogicalArray(mask) if mask.data == vec![0, 1]));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ismissing_preserves_rank_zero_and_rank_one_numeric_shapes() {
    let cases = [
        Value::Tensor(Tensor::new(vec![f64::NAN], Vec::new()).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, f64::NAN], vec![2]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(f64::NAN, 0.0)], Vec::new()).unwrap()),
    ];
    for (input, expected_shape, expected_mask) in [
        (cases[0].clone(), Vec::new(), vec![1]),
        (cases[1].clone(), vec![2], vec![0, 1]),
        (cases[2].clone(), Vec::new(), vec![1]),
    ] {
        let result = block_on(ismissing_builtin(input)).expect("ismissing numeric shape");
        assert!(matches!(
            result,
            Value::LogicalArray(mask)
                if mask.shape == expected_shape && mask.data == expected_mask
        ));
    }
}

#[test]
fn ismissing_typed_integer_tensor_ignores_f64_mirror() {
    let input = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![1, 3])
        .expect("integer tensor");

    let result = block_on(ismissing_builtin(Value::Tensor(input))).unwrap();

    assert!(matches!(
        result,
        Value::LogicalArray(mask) if mask.data == vec![0, 0, 0] && mask.shape == vec![1, 3]
    ));
}

#[test]
fn ismissing_returns_same_shaped_false_for_all_integer_classes() {
    let storages = [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
        IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
        IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
        IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
    ];
    for storage in storages {
        let input = Tensor::new_integer(storage, vec![2, 1]).expect("integer tensor");
        let result = block_on(ismissing_builtin(Value::Tensor(input))).expect("ismissing");
        assert!(matches!(
            result,
            Value::LogicalArray(mask)
                if mask.shape == vec![2, 1] && mask.data == vec![0, 0]
        ));
    }
}

#[test]
fn ismissing_resident_integer_uses_shape_metadata_and_returns_host_mask() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX]), vec![1, 2])
            .expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
            .expect("resident ismissing extension");
        assert!(matches!(
            result,
            Value::LogicalArray(mask)
                if mask.shape == vec![1, 2] && mask.data == vec![0, 0]
        ));
        assert!(gpu_helpers::exact_provider_for_handle(&handle).is_some());
        handle.release().unwrap();
    });
}

#[test]
fn ismissing_integer_owner_rejects_logical_reinterpretation() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::I8(vec![1]), vec![1, 1]).expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let before = handle.wire_descriptor();
        assert!(handle
            .with_view(handle.view().clone().with_logical(true))
            .is_err());
        assert_eq!(handle.wire_descriptor(), before);
        let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
            .expect("rejected view must leave original integer input valid");
        assert!(
            matches!(result, Value::LogicalArray(mask) if mask.shape == vec![1, 1] && mask.data == vec![0])
        );
        let preserved = test_support::gather(Value::GpuTensor(handle.clone())).unwrap();
        assert_eq!(
            preserved.integer_storage(),
            Some(&IntegerStorage::I8(vec![1]))
        );
        handle.release().unwrap();
    });
}

#[test]
fn ismissing_matlab_mode_only_gates_explicit_resident_input() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let result = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                .expect("automatic residency is transparent");
            assert!(matches!(
                result,
                Value::LogicalArray(mask)
                    if mask.shape == vec![1, 1] && mask.data == vec![0]
            ));
        }
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(ismissing_builtin(Value::GpuTensor(handle.clone())))
                .expect_err("explicit resident input is a compatibility-gated extension");
            assert_eq!(
                error.identifier(),
                ISMISSING_RESIDENT_INPUT_EXTENSION.error_identifier
            );
        }
        handle.release().unwrap();
    });
}

#[test]
fn anymissing_returns_false_for_every_integer_storage_class() {
    let storages = [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![u8::MIN, u8::MAX]),
        IntegerStorage::U16(vec![u16::MIN, u16::MAX]),
        IntegerStorage::U32(vec![u32::MIN, u32::MAX]),
        IntegerStorage::U64(vec![u64::MIN, u64::MAX]),
    ];
    for storage in storages {
        let input = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        assert_eq!(
            block_on(anymissing_builtin(Value::Tensor(input))).unwrap(),
            Value::Bool(false)
        );
    }
}
