use super::*;

#[test]
fn encoding_error_identifiers_follow_typed_owner_descriptors() {
    for (owner, invalid_id, internal_id) in [
        (
            EncodingBuiltin::Dummyvar,
            "RunMat:dummyvar:InvalidArgument",
            "RunMat:dummyvar:Internal",
        ),
        (
            EncodingBuiltin::OnehotEncode,
            "RunMat:onehotencode:InvalidArgument",
            "RunMat:onehotencode:Internal",
        ),
        (
            EncodingBuiltin::OnehotDecode,
            "RunMat:onehotdecode:InvalidArgument",
            "RunMat:onehotdecode:Internal",
        ),
    ] {
        assert_eq!(
            invalid(owner, "invalid").identifier.as_deref(),
            Some(invalid_id)
        );
        assert_eq!(
            internal(owner, "internal").identifier.as_deref(),
            Some(internal_id)
        );
    }
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

#[test]
fn categorical_classes_preserve_unused_categories() {
    let categories = Value::StringArray(
        StringArray::new(
            vec!["low".into(), "medium".into(), "high".into()],
            vec![1, 3],
        )
        .unwrap(),
    );
    let group = crate::builtins::table::categorical_from_args(
        crate::call::FunctionArgumentView::from_slice(&[
            Value::StringArray(
                StringArray::new(vec!["high".into(), "low".into()], vec![2, 1]).unwrap(),
            ),
            categories.clone(),
        ]),
    )
    .unwrap();
    let Value::Tensor(dummy) = block_on(dummyvar_builtin(group.clone())).unwrap() else {
        panic!("dummy tensor");
    };
    assert_eq!(dummy.shape(), &[2, 3]);
    assert_eq!(dummy.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

    let Value::Tensor(encoded) = block_on(onehotencode_builtin(
        group,
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap() else {
        panic!("encoded tensor");
    };
    assert_eq!(encoded.shape(), &[2, 3]);
    assert_eq!(
        encoded.materialize_f64(),
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
}

#[test]
fn onehot_feature_dimension_parses_integer_bounds_exactly() {
    assert_eq!(
        positive_dim(EncodingBuiltin::OnehotEncode, &Value::Int(IntValue::U16(2))).unwrap(),
        2
    );
    assert_eq!(
        positive_dim(
            EncodingBuiltin::OnehotEncode,
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
        assert!(positive_dim(EncodingBuiltin::OnehotEncode, &value).is_err());
    }
}

#[test]
fn onehot_output_type_name_value_aliases_are_mode_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let encode_error = block_on(onehotencode_builtin(
        int_tensor(IntegerStorage::U8(vec![1, 2]), vec![2, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            int_tensor(IntegerStorage::U8(vec![1, 2]), vec![1, 2]),
            Value::from("OutputType"),
            Value::from("uint8"),
        ]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("OutputType"),
            Value::from("uint8"),
        ]),
    ))
    .expect_err("onehotdecode alias gate");
    assert_eq!(
        decode_error.identifier(),
        ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn rejects_bad_dimensions_and_unknown_labels() {
    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, 2.0], vec![1, 2]),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("singleton"));

    let err = block_on(onehotencode_builtin(
        tensor(vec![1.0], vec![1, 1]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("ClassNames"),
            tensor(vec![1.0, f64::NAN], vec![1, 2]),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("cannot contain missing"));

    let err = block_on(onehotdecode_builtin(
        tensor(vec![1.2, 0.0], vec![1, 2]),
        tensor(vec![1.0, 2.0], vec![1, 2]),
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .unwrap_err();
    assert!(err.message.contains("between 0 and 1"));
}
