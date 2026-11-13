use super::*;

#[test]
fn known_later_invalid_option_precedes_deferred_control_transfer() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let deferred_output_type = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(onehotencode_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            deferred_output_type,
            Value::from("NotAnEncodingOption"),
            Value::from("double"),
        ]),
    ))
    .expect_err("known invalid option must precede earlier deferred control gather");
    assert!(error.message.contains("unsupported option"), "{error}");
    assert_eq!(
        error.identifier.as_deref(),
        Some("RunMat:onehotencode:InvalidArgument")
    );
}

#[test]
fn onehotencode_host_invalid_controls_precede_resident_input_access() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![2, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let error = block_on(onehotencode_builtin(
        resident,
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("UnsupportedOption"),
            Value::from("double"),
        ]),
    ))
    .expect_err("invalid host option must precede input provider access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:onehotencode:InvalidArgument")
    );
    assert!(error.message.contains("unsupported option"));
}

#[test]
fn onehotencode_uses_feature_dimension_and_class_names() {
    let Value::LogicalArray(out) = block_on(onehotencode_builtin(
        Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
        ),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("logical"),
            Value::from("ClassNames"),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap() else {
        panic!("logical");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.data, vec![0, 1, 0, 1, 0, 1]);
}

#[test]
fn onehotencode_accepts_typename_before_name_value_options() {
    let Value::Tensor(out) = block_on(onehotencode_builtin(
        Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
        ),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("single"),
            Value::from("ClassNames"),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn onehotencode_requires_classnames_for_noncategorical_and_marks_excluded_labels() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("ClassNames is required"));

    let Value::Tensor(out) = block_on(onehotencode_builtin(
        tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ]),
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 2]);
    assert_f64_slice_eq_nan(
        &out.materialize_f64(),
        &[1.0, f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN],
    );

    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 3.0], vec![2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("logical"),
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("logical output cannot represent"));
}

#[test]
fn onehotencode_reads_typed_integer_labels_and_classes_exactly() {
    let Value::Tensor(out) = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U16(vec![1, 2, 1]), vec![3, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::I16(vec![1, 2]), vec![1, 2]),
        ]),
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn onehotencode_distinguishes_adjacent_wide_integer_labels() {
    let lower = (1_u64 << 53) + 1;
    let upper = lower + 1;
    let Value::Tensor(out) = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U64(vec![upper, lower, upper]), vec![3, 1]),
        Value::Int(IntValue::U8(2)),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("uint8"),
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::U64(vec![lower, upper]), vec![1, 2]),
        ]),
    ))
    .expect("wide labels") else {
        panic!("typed one-hot tensor")
    };
    assert_eq!(out.numeric_dtype(), NumericDType::U8);
    assert_eq!(
        out.integer_storage(),
        Some(&IntegerStorage::U8(vec![0, 1, 0, 1, 0, 1]))
    );
}

#[test]
fn onehotencode_constructs_every_requested_native_numeric_class() {
    let labels =
        Value::StringArray(StringArray::new(vec!["b".into(), "a".into()], vec![2, 1]).unwrap());
    let classes =
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap());
    for (typename, dtype) in [
        ("double", NumericDType::F64),
        ("single", NumericDType::F32),
        ("int8", NumericDType::I8),
        ("int16", NumericDType::I16),
        ("int32", NumericDType::I32),
        ("int64", NumericDType::I64),
        ("uint8", NumericDType::U8),
        ("uint16", NumericDType::U16),
        ("uint32", NumericDType::U32),
        ("uint64", NumericDType::U64),
    ] {
        let Value::Tensor(out) = block_on(onehotencode_builtin(
            labels.clone(),
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from(typename),
                Value::from("ClassNames"),
                classes.clone(),
            ]),
        ))
        .expect("numeric typename") else {
            panic!("numeric tensor")
        };
        assert_eq!(out.numeric_dtype(), dtype, "{typename}");
        assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
    }
}

#[test]
fn onehotencode_integer_output_rejects_missing_or_excluded_labels() {
    let error = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::I64(vec![1, 3]), vec![2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("uint64"),
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::I64(vec![1, 2]), vec![1, 2]),
        ]),
    ))
    .expect_err("integer output cannot hold NaN vectors");
    assert!(error
        .message()
        .contains("cannot represent missing or excluded"));
}

#[test]
fn onehotencode_rejects_pathological_feature_dimension_before_resizing_shape() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0], vec![1, 1]),
        Value::Num(1_000_000_000.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("featureDim is too large"));
}
