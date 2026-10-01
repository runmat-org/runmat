use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage};

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn class_identity_is_typed_and_exact_across_numeric_and_text_labels() {
    assert_eq!(
        label_key(&Label::Numeric(NumericScalar::F64(-0.0))),
        label_key(&Label::Numeric(NumericScalar::from(IntValue::U64(0))))
    );
    assert_eq!(
        label_key(&Label::Numeric(NumericScalar::F64(7.0))),
        label_key(&Label::Numeric(NumericScalar::from(IntValue::I64(7))))
    );
    assert_ne!(
        label_key(&Label::Numeric(NumericScalar::from(IntValue::U64(
            9_007_199_254_740_993,
        )))),
        label_key(&Label::Numeric(NumericScalar::from(IntValue::U64(
            9_007_199_254_740_992,
        ))))
    );
    assert_ne!(
        label_key(&Label::Text("n:i:7".into())),
        label_key(&Label::Numeric(NumericScalar::F64(7.0)))
    );
    assert_ne!(
        label_key(&Label::Logical(true)),
        label_key(&Label::Numeric(NumericScalar::F64(1.0)))
    );
}

fn mirrorless_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn assert_f64_slice_eq_nan(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a.is_nan() && b.is_nan()) || (*a - *b).abs() < 1.0e-12,
            "mismatch at {idx}: {a:?} != {b:?}"
        );
    }
}

#[test]
fn dummyvar_encodes_numeric_columns_and_missing() {
    let Value::Tensor(out) = block_on(dummyvar_builtin(tensor(
        vec![1.0, 2.0, f64::NAN, 1.0, 2.0, 2.0],
        vec![3, 2],
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 4]);
    assert_f64_slice_eq_nan(
        &out.materialize_f64(),
        &[
            1.0,
            0.0,
            f64::NAN,
            0.0,
            1.0,
            f64::NAN,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ],
    );
}

#[test]
fn dummyvar_encodes_cell_grouping_variables() {
    let phone = Value::Cell(
        CellArray::new(
            vec![
                Value::from("mobile"),
                Value::from("landline"),
                Value::from("mobile"),
            ],
            3,
            1,
        )
        .unwrap(),
    );
    let region = tensor(vec![2.0, 1.0, 2.0], vec![3, 1]);
    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(vec![phone, region], 1, 2).unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn dummyvar_treats_row_cellstr_as_one_grouping_vector() {
    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(
            vec![Value::from("red"), Value::from("blue"), Value::from("red")],
            1,
            3,
        )
        .unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn dummyvar_treats_numeric_row_vector_as_one_grouping_vector() {
    let Value::Tensor(out) =
        block_on(dummyvar_builtin(tensor(vec![1.0, 2.0, 1.0], vec![1, 3]))).unwrap()
    else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn dummyvar_reads_typed_integer_labels_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::Tensor(out) = block_on(dummyvar_builtin(int_tensor(
        IntegerStorage::I16(vec![1, 2, 1, 2, 2, 1]),
        vec![3, 2],
    )))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn dummyvar_supports_all_integer_classes_in_extension_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let storages = [
        runmat_value::IntegerStorage::I8(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I16(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I32(vec![1, 2, 1]),
        runmat_value::IntegerStorage::I64(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U8(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U16(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U32(vec![1, 2, 1]),
        runmat_value::IntegerStorage::U64(vec![1, 2, 1]),
    ];
    for storage in storages {
        let Value::Tensor(output) =
            block_on(dummyvar_builtin(int_tensor(storage, vec![3, 1]))).expect("integer groups")
        else {
            panic!("expected double tensor")
        };
        assert!(output.integer_storage().is_none());
        assert_eq!(output.shape, vec![3, 2]);
        assert_eq!(output.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }
}

#[test]
fn dummyvar_rejects_wide_typed_levels_before_floating_conversion() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in [
        IntegerStorage::I64(vec![(1_i64 << 53) + 1]),
        IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
        IntegerStorage::U64(vec![u64::MAX]),
    ] {
        let error = block_on(dummyvar_builtin(int_tensor(storage, vec![1, 1])))
            .expect_err("wide typed labels cannot request an unbounded design matrix");
        assert!(error.message().contains("output is too large"));
    }

    let error = block_on(dummyvar_builtin(int_tensor(
        IntegerStorage::I64(vec![i64::MIN]),
        vec![1, 1],
    )))
    .expect_err("negative typed labels remain invalid");
    assert!(error.message().contains("positive integer levels"));
}

#[test]
fn dummyvar_integer_extension_rejects_before_computation_in_strict_mode() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(dummyvar_builtin(int_tensor(
        runmat_value::IntegerStorage::U8(vec![1, 2]),
        vec![2, 1],
    )))
    .expect_err("integer extension gate");
    assert_eq!(
        error.identifier(),
        DUMMYVAR_INTEGER_GROUP_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn dummyvar_gpu_extension_rejects_before_provider_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
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
    let error = block_on(dummyvar_builtin(resident)).expect_err("GPU extension gate");
    assert_eq!(
        error.identifier(),
        DUMMYVAR_GPU_GROUP_EXTENSION.error_identifier
    );
    drop(strict);
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
        vec![Value::from("UnsupportedOption"), Value::from("double")],
    ))
    .expect_err("invalid host option must precede input provider access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:onehotencode:InvalidArgument")
    );
    assert!(error.message.contains("unsupported option"));
}

#[test]
fn onehotdecode_host_invalid_dimension_precedes_resident_input_access() {
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
    let error = block_on(onehotdecode_builtin(
        resident,
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(0.0),
        vec![],
    ))
    .expect_err("invalid host dimension must precede input provider access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:onehotdecode:InvalidArgument")
    );
    assert!(error
        .message
        .contains("featureDim must be a positive integer"));
}

#[test]
fn dummyvar_rejects_unrepresentable_numeric_level_before_cast() {
    let err = block_on(dummyvar_builtin(tensor(
        vec![usize::MAX as f64],
        vec![1, 1],
    )))
    .unwrap_err();
    assert!(err.message.contains("positive integer levels"));
}

#[test]
fn dummyvar_grouping_detection_uses_typed_integer_storage_len() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let first = mirrorless_int_tensor(IntegerStorage::I16(vec![1, 2, 1]), vec![3, 1]);
    let second = mirrorless_int_tensor(IntegerStorage::I16(vec![2, 2, 1]), vec![3, 1]);

    let Value::Tensor(out) = block_on(dummyvar_builtin(Value::Cell(
        CellArray::new(vec![first, second], 1, 2).unwrap(),
    )))
    .unwrap() else {
        panic!("tensor");
    };

    assert_eq!(out.shape, vec![3, 4]);
    assert_eq!(
        out.materialize_f64(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn dummyvar_rejects_huge_numeric_level_before_allocating_classes() {
    let err = block_on(dummyvar_builtin(tensor(vec![1_000_000_000.0], vec![1, 1]))).unwrap_err();
    assert!(err.message.contains("output is too large"));
}

#[test]
fn categorical_classes_preserve_unused_categories() {
    let categories = Value::StringArray(
        StringArray::new(
            vec!["low".into(), "medium".into(), "high".into()],
            vec![1, 3],
        )
        .unwrap(),
    );
    let group = crate::builtins::table::categorical_from_args(vec![
        Value::StringArray(
            StringArray::new(vec!["high".into(), "low".into()], vec![2, 1]).unwrap(),
        ),
        categories.clone(),
    ])
    .unwrap();
    let Value::Tensor(dummy) = block_on(dummyvar_builtin(group.clone())).unwrap() else {
        panic!("dummy tensor");
    };
    assert_eq!(dummy.shape, vec![2, 3]);
    assert_eq!(dummy.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

    let Value::Tensor(encoded) =
        block_on(onehotencode_builtin(group, Value::Num(2.0), Vec::new())).unwrap()
    else {
        panic!("encoded tensor");
    };
    assert_eq!(encoded.shape, vec![2, 3]);
    assert_eq!(
        encoded.materialize_f64(),
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn onehotencode_uses_feature_dimension_and_class_names() {
    let Value::LogicalArray(out) = block_on(onehotencode_builtin(
        Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
        ),
        Value::Num(2.0),
        vec![
            Value::from("logical"),
            Value::from("ClassNames"),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ],
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
        vec![
            Value::from("single"),
            Value::from("ClassNames"),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn onehotencode_requires_classnames_for_noncategorical_and_marks_excluded_labels() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::Num(2.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message.contains("ClassNames is required"));

    let Value::Tensor(out) = block_on(onehotencode_builtin(
        tensor(vec![1.0, f64::NAN, 3.0], vec![3, 1]),
        Value::Num(2.0),
        vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ],
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_f64_slice_eq_nan(
        &out.materialize_f64(),
        &[1.0, f64::NAN, f64::NAN, 0.0, f64::NAN, f64::NAN],
    );

    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 3.0], vec![2, 1]),
        Value::Num(2.0),
        vec![
            Value::from("logical"),
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("logical output cannot represent"));
}

#[test]
fn onehotencode_reads_typed_integer_labels_and_classes_exactly() {
    let Value::Tensor(out) = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U16(vec![1, 2, 1]), vec![3, 1]),
        Value::Num(2.0),
        vec![
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::I16(vec![1, 2]), vec![1, 2]),
        ],
    ))
    .unwrap() else {
        panic!("tensor");
    };
    assert_eq!(out.shape, vec![3, 2]);
    assert_eq!(out.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn onehotencode_distinguishes_adjacent_wide_integer_labels() {
    let lower = (1_u64 << 53) + 1;
    let upper = lower + 1;
    let Value::Tensor(out) = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U64(vec![upper, lower, upper]), vec![3, 1]),
        Value::Int(IntValue::U8(2)),
        vec![
            Value::from("uint8"),
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::U64(vec![lower, upper]), vec![1, 2]),
        ],
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
            vec![
                Value::from(typename),
                Value::from("ClassNames"),
                classes.clone(),
            ],
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
        vec![
            Value::from("uint64"),
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::I64(vec![1, 2]), vec![1, 2]),
        ],
    ))
    .expect_err("integer output cannot hold NaN vectors");
    assert!(error
        .message()
        .contains("cannot represent missing or excluded"));
}

#[test]
fn onehot_feature_dimension_parses_integer_bounds_exactly() {
    assert_eq!(
        positive_dim("onehotencode", &Value::Int(IntValue::U16(2))).unwrap(),
        2
    );
    assert_eq!(
        positive_dim(
            "onehotencode",
            &mirrorless_int_tensor(IntegerStorage::U64(vec![2]), vec![1, 1])
        )
        .unwrap(),
        2
    );

    for value in [
        Value::Int(IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64),
        Value::Num(usize::MAX as f64 + 1.0),
        mirrorless_int_tensor(IntegerStorage::U64(vec![0]), vec![1, 1]),
    ] {
        assert!(positive_dim("onehotencode", &value).is_err());
    }
}

#[test]
fn onehotencode_rejects_pathological_feature_dimension_before_resizing_shape() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0], vec![1, 1]),
        Value::Num(1_000_000_000.0),
        vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("featureDim is too large"));
}

#[test]
fn onehotdecode_round_trips_string_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let encoded =
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 1, 1, 0, 0], vec![3, 2]).unwrap());
    let Value::StringArray(out) = block_on(onehotdecode_builtin(
        encoded,
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::Num(2.0),
        vec![Value::from("string")],
    ))
    .unwrap() else {
        panic!("strings");
    };
    assert_eq!(out.shape, vec![3, 1]);
    assert_eq!(out.data, vec!["b", "a", "a"]);
}

#[test]
fn onehotdecode_returns_categorical_for_categorical_classes() {
    let categories = Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
    );
    let classes = crate::builtins::table::categorical_from_args(vec![
        Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
        ),
        categories,
    ])
    .unwrap();
    let encoded = Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap());
    let Value::Object(out) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(1.0),
        Vec::new(),
    ))
    .unwrap() else {
        panic!("categorical");
    };
    assert!(out.is_class(runmat_types::standard::CATEGORICAL));
    let labels = crate::builtins::table::categorical_labels(&Value::Object(out)).unwrap();
    assert_eq!(labels, vec!["high", "low"]);
}

#[test]
fn onehotdecode_defaults_to_categorical_and_preserves_class_order() {
    let encoded = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap());
    let classes = Value::StringArray(
        StringArray::new(
            vec!["low".into(), "high".into(), "unused".into()],
            vec![1, 3],
        )
        .unwrap(),
    );
    let err = block_on(onehotdecode_builtin(
        encoded.clone(),
        classes.clone(),
        Value::Num(2.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message.contains("classes length"));

    let classes = Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
    );
    let Value::Object(out) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(2.0),
        Vec::new(),
    ))
    .unwrap() else {
        panic!("categorical");
    };
    let categories = match out.properties.get("Categories") {
        Some(Value::StringArray(array)) => array.data.clone(),
        other => panic!("categories {other:?}"),
    };
    assert_eq!(categories, vec!["low", "high"]);
}

#[test]
fn onehotdecode_reads_typed_integer_encoded_array_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let encoded = int_tensor(IntegerStorage::U8(vec![0, 1, 1, 0]), vec![2, 2]);
    let Value::StringArray(out) = block_on(onehotdecode_builtin(
        encoded,
        Value::StringArray(
            StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
        ),
        Value::Num(2.0),
        vec![Value::from("string")],
    ))
    .unwrap() else {
        panic!("strings");
    };
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(out.data, vec!["high", "low"]);
}

#[test]
fn onehotdecode_preserves_wide_integer_classes_and_native_output_types() {
    let lower = (1_u64 << 53) + 1;
    let upper = lower + 1;
    let encoded = Value::Tensor(
        Tensor::from_numeric_storage(NumericStorage::F32(vec![0.0, 1.0, 1.0, 0.0]), vec![2, 2])
            .unwrap(),
    );
    let classes = int_tensor(IntegerStorage::U64(vec![lower, upper]), vec![1, 2]);

    let Value::Tensor(decoded) = block_on(onehotdecode_builtin(
        encoded.clone(),
        classes.clone(),
        Value::Int(IntValue::U8(2)),
        vec![Value::from("uint64")],
    ))
    .expect("wide uint64 output") else {
        panic!("typed output")
    };
    assert_eq!(decoded.numeric_dtype(), NumericDType::U64);
    assert_eq!(
        decoded.integer_storage(),
        Some(&IntegerStorage::U64(vec![upper, lower]))
    );

    let Value::Object(categorical) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(2.0),
        Vec::new(),
    ))
    .expect("categorical wide labels") else {
        panic!("categorical output")
    };
    assert_eq!(
        crate::builtins::table::categorical_labels(&Value::Object(categorical)).unwrap(),
        vec![upper.to_string(), lower.to_string()]
    );
}

#[test]
fn onehot_output_type_name_value_aliases_are_mode_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let encode_error = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U8(vec![1, 2]), vec![2, 1]),
        Value::Num(2.0),
        vec![
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
            Value::from("OutputType"),
            Value::from("uint8"),
        ],
    ))
    .expect_err("onehotencode alias gate");
    assert_eq!(
        encode_error.identifier(),
        ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION.error_identifier
    );

    let decode_error = block_on(onehotdecode_builtin(
        tensor(vec![1.0, 0.0], vec![2, 1]),
        int_tensor(IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
        Value::Num(1.0),
        vec![Value::from("OutputType"), Value::from("uint8")],
    ))
    .expect_err("onehotdecode alias gate");
    assert_eq!(
        decode_error.identifier(),
        ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn onehotdecode_integer_probabilities_are_mode_gated_before_computation() {
    let encoded = int_tensor(IntegerStorage::U8(vec![0, 1, 1, 0]), vec![2, 2]);
    let classes = Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
    );
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(onehotdecode_builtin(
        encoded.clone(),
        classes.clone(),
        Value::Num(2.0),
        vec![Value::from("string")],
    ))
    .expect_err("strict mode rejects integer probabilities");
    assert_eq!(
        error.identifier(),
        ONEHOTDECODE_BROAD_ENCODED_EXTENSION.error_identifier
    );
    drop(strict);

    let extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::StringArray(decoded) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(2.0),
        vec![Value::from("string")],
    ))
    .expect("RunMat extension") else {
        panic!("string output")
    };
    assert_eq!(decoded.data, vec!["high", "low"]);
    drop(extension);
}

#[test]
fn onehotdecode_categorical_output_allows_repeated_decoded_labels() {
    let encoded =
        Value::Tensor(Tensor::new(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0], vec![3, 2]).unwrap());
    let classes = Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
    );
    let Value::Object(out) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(2.0),
        vec![Value::from("categorical")],
    ))
    .unwrap() else {
        panic!("categorical");
    };
    let labels = crate::builtins::table::categorical_labels(&Value::Object(out)).unwrap();
    assert_eq!(labels, vec!["low", "low", "low"]);
}

#[test]
fn rejects_bad_dimensions_and_unknown_labels() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::Num(1.0),
        vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("singleton"));

    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0], vec![1, 1]),
        Value::Num(2.0),
        vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, f64::NAN], vec![1, 2]),
        ],
    ))
    .unwrap_err();
    assert!(err.message.contains("cannot contain missing"));

    let err = block_on(onehotdecode_builtin(
        tensor(vec![1.2, 0.0], vec![1, 2]),
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        Vec::new(),
    ))
    .unwrap_err();
    assert!(err.message.contains("between 0 and 1"));
}
