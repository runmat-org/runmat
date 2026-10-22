use super::*;

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
        crate::builtins::common::test_support::builtin_arguments(vec![]),
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
fn onehotdecode_round_trips_string_classes() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let encoded =
        Value::LogicalArray(LogicalArray::new(vec![0, 1, 1, 1, 0, 0], vec![3, 2]).unwrap());
    let Value::StringArray(out) = block_on(onehotdecode_builtin(
        encoded,
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("string")]),
    ))
    .unwrap() else {
        panic!("strings");
    };
    assert_eq!(out.shape, vec![3, 1]);
    assert_eq!(out.data(), vec!["b", "a", "a"]);
}

#[test]
fn onehotdecode_returns_categorical_for_categorical_classes() {
    let categories = Value::StringArray(
        StringArray::new(vec!["low".into(), "high".into()], vec![1, 2]).unwrap(),
    );
    let classes = crate::builtins::table::categorical_from_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::StringArray(
                StringArray::new(vec!["low".into(), "high".into()], vec![2, 1]).unwrap(),
            ),
            categories,
        ]),
    )
    .unwrap();
    let encoded = Value::Tensor(Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).unwrap());
    let Value::Object(out) = block_on(onehotdecode_builtin(
        encoded,
        classes,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap() else {
        panic!("categorical");
    };
    let categories = match out.properties().get("Categories") {
        Some(Value::StringArray(array)) => array.data(),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("string")]),
    ))
    .unwrap() else {
        panic!("strings");
    };
    assert_eq!(out.shape, vec![2, 1]);
    assert_eq!(out.data(), vec!["high", "low"]);
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("uint64")]),
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
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("string")]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("string")]),
    ))
    .expect("RunMat extension") else {
        panic!("string output")
    };
    assert_eq!(decoded.data(), vec!["high", "low"]);
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("categorical")]),
    ))
    .unwrap() else {
        panic!("categorical");
    };
    let labels = crate::builtins::table::categorical_labels(&Value::Object(out)).unwrap();
    assert_eq!(labels, vec!["low", "low", "low"]);
}
