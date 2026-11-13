use super::*;
use runmat_value::ObjectInstance;

#[test]
fn preferences_round_trip() {
    let runtime = preference_context();
    run(setpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("runmatTest".to_string().into()),
            Value::String("answer".to_string().into()),
            Value::Num(42.0),
        ]),
    ))
    .unwrap();
    assert_eq!(
        run(ispref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("runmatTest".to_string().into()),
                Value::String("answer".to_string().into())
            ])
        ))
        .unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("runmatTest".to_string().into()),
                Value::String("answer".to_string().into())
            ])
        ))
        .unwrap(),
        Value::Num(42.0)
    );
}

#[test]
fn preferences_require_an_installed_application_service_and_isolate_explicit_stores() {
    let plain = RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let group = Value::String("isolatedPreferenceStore".into());
    let name = Value::String("value".into());
    let error = run(setpref_builtin(
        Some(plain),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            name.clone(),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("no ambient preference fallback");
    assert_eq!(
        error.identifier(),
        Some("RunMat:RuntimeContext:CapabilityUnavailable")
    );

    let first = preference_context();
    let second = preference_context();
    run(setpref_builtin(
        Some(first.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            name.clone(),
            Value::Num(7.0),
        ]),
    ))
    .expect("first isolated store");
    assert_eq!(
        run(ispref_builtin(
            Some(second),
            crate::builtins::common::test_support::builtin_arguments(vec![
                group.clone(),
                name.clone()
            ])
        ))
        .expect("second isolated store"),
        Value::Bool(false)
    );
    assert_eq!(
        run(getpref_builtin(
            Some(first),
            crate::builtins::common::test_support::builtin_arguments(vec![group, name])
        ))
        .expect("first store retains value"),
        Value::Num(7.0)
    );
}

#[test]
fn multi_name_setpref_and_ispref_keep_shape_and_missing_positions() {
    let runtime = preference_context();
    let group = Value::String("runmatMultiPreferenceForms".into());
    let names = Value::StringArray(
        runmat_value::StringArray::new(vec!["first".into(), "second".into()], vec![2, 1])
            .expect("names shape"),
    );
    let values = Value::Cell(
        CellArray::new_with_shape(
            vec![
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
                Value::Num(2.0),
            ],
            vec![2, 1],
        )
        .expect("values shape"),
    );
    assert_eq!(
        run(setpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                group.clone(),
                names.clone(),
                values.clone()
            ])
        ))
        .expect("set both"),
        Value::Num(0.0)
    );
    let queried = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            names.clone(),
        ]),
    ))
    .expect("retrieve both");
    assert_eq!(queried, values);
    let flags = run(ispref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![group.clone(), names]),
    ))
    .expect("query both");
    let Value::LogicalArray(flags) = flags else {
        panic!("logical array");
    };
    assert_eq!(flags.shape, vec![2, 1]);
    assert_eq!(flags.data.as_slice(), &[1, 1]);

    let mixed = Value::StringArray(
        runmat_value::StringArray::new(vec!["first".into(), "missing".into()], vec![1, 2])
            .expect("mixed shape"),
    );
    let flags = run(ispref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![group, mixed]),
    ))
    .expect("mixed query");
    let Value::LogicalArray(flags) = flags else {
        panic!("logical array");
    };
    assert_eq!(flags.shape, vec![1, 2]);
    assert_eq!(flags.data.as_slice(), &[1, 0]);
}

#[test]
fn getpref_persists_and_preserves_every_integer_scalar_class() {
    let runtime = preference_context();
    for (index, value) in [
        Value::Int(runmat_value::IntValue::I8(i8::MIN)),
        Value::Int(runmat_value::IntValue::I16(i16::MIN)),
        Value::Int(runmat_value::IntValue::I32(i32::MIN)),
        Value::Int(runmat_value::IntValue::I64(i64::MIN)),
        Value::Int(runmat_value::IntValue::U8(u8::MAX)),
        Value::Int(runmat_value::IntValue::U16(u16::MAX)),
        Value::Int(runmat_value::IntValue::U32(u32::MAX)),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    ]
    .into_iter()
    .enumerate()
    {
        let preference = format!("scalar{index}");
        let created = run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference.clone().into()),
                value.clone(),
            ]),
        ))
        .expect("create integer preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference.into()),
            ]),
        ))
        .expect("read integer preference");
        assert_eq!(stored, value);
    }
}

#[test]
fn getpref_preserves_every_integer_array_class_and_shape() {
    let runtime = preference_context();
    for (index, storage) in [
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
    ]
    .into_iter()
    .enumerate()
    {
        let value = Value::Tensor(
            Tensor::new_integer(storage, vec![1, 2]).expect("integer preference array"),
        );
        let preference = format!("array{index}");
        let created = run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference.clone().into()),
                value.clone(),
            ]),
        ))
        .expect("create integer array preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference.into()),
            ]),
        ))
        .expect("read integer array preference");
        assert_eq!(stored, value);
    }
}

#[test]
fn getpref_existing_value_wins_over_default_without_conversion() {
    let runtime = preference_context();
    let existing = Value::Int(runmat_value::IntValue::U64(u64::MAX));
    run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            existing.clone(),
        ]),
    ))
    .expect("create preference");
    let result = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            Value::FunctionHandle("session_function".into()),
        ]),
    ))
    .expect("ignored unsupported host default must not narrow existing form");
    assert_eq!(result, existing);
}

#[test]
fn getpref_preserves_arbitrary_object_values_and_zero_argument_structure() {
    let runtime = preference_context();
    let mut object = ObjectInstance::new("PreferencePayload".to_string());
    object.replace_property_unbounded(
        "revision".to_string(),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    );
    let value = Value::Object(object);
    let created = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
            value.clone(),
        ]),
    ))
    .expect("create object preference");
    assert_eq!(created, value);
    let stored = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
        ]),
    ))
    .expect("read object preference");
    assert_eq!(stored, value);

    let all = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("all preferences");
    let Value::Struct(all) = all else {
        panic!("expected preference structure");
    };
    assert!(all.fields.contains_key("getprefObjects"));
}

#[test]
fn getpref_multiple_names_preserve_query_shape_and_persist_defaults() {
    let runtime = preference_context();
    let names = Value::StringArray(
        runmat_value::StringArray::new(vec!["signed".into(), "unsigned".into()], vec![2, 1])
            .expect("preference names"),
    );
    let defaults = Value::Cell(
        CellArray::new_with_shape(
            vec![
                Value::Int(runmat_value::IntValue::I64(i64::MIN)),
                Value::Int(runmat_value::IntValue::U64(u64::MAX)),
            ],
            vec![2, 1],
        )
        .expect("defaults"),
    );
    let created = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefMultiple".into()),
            names.clone(),
            defaults.clone(),
        ]),
    ))
    .expect("create multiple preferences");
    assert_eq!(created, defaults);
    let stored = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefMultiple".into()),
            names,
        ]),
    ))
    .expect("read multiple preferences");
    assert_eq!(stored, defaults);

    let cellstr = Value::Cell(
        CellArray::new_with_shape(
            vec![char_value("signed"), char_value("unsigned")],
            vec![1, 2],
        )
        .expect("cellstr names"),
    );
    let stored = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("getprefMultiple".into()),
            cellstr,
        ]),
    ))
    .expect("read cellstr preferences");
    let Value::Cell(stored) = stored else {
        panic!("expected cell result");
    };
    assert_eq!(stored.shape, vec![1, 2]);
}
