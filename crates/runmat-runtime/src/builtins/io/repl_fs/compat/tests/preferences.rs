use super::*;
use crate::builtins::io::repl_fs::preferences::{getpref_builtin, ispref_builtin, setpref_builtin};
use crate::context::{IsolatedMemoryPreferenceService, RuntimeContext, RuntimeServicePorts};
use runmat_builtins::{
    GETPREF_DESCRIPTOR, GETPREF_GROUP_QUERY_EXTENSION, GETPREF_INTEGER_CAPABILITIES,
};

fn preference_context() -> RuntimeContext {
    RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default()
            .with_preferences(std::rc::Rc::new(IsolatedMemoryPreferenceService::new())),
    )
}

#[test]
fn preference_base_forms_use_matlab_identity_with_explicit_runmat_extensions() {
    for name in ["getpref", "ispref", "setpref"] {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).expect("catalog entry");
        assert_eq!(
            entry.contract.compatibility,
            runmat_builtins::BuiltinCompatibility::Matlab
        );
    }
    assert_eq!(
        runmat_builtins::GETPREF_GROUP_QUERY_EXTENSION.mode,
        runmat_builtins::BuiltinExtensionMode::RunMatOnly
    );
    assert_eq!(
        runmat_builtins::SETPREF_STATUS_EXTENSION.mode,
        runmat_builtins::BuiltinExtensionMode::RunMatOnly
    );
}

#[test]
fn ispref_rejects_integer_and_resident_controls_without_gather() {
    let runtime = preference_context();
    for integer in [
        runmat_value::IntValue::I8(-1),
        runmat_value::IntValue::I16(-2),
        runmat_value::IntValue::I32(-3),
        runmat_value::IntValue::I64(i64::MIN),
        runmat_value::IntValue::U8(1),
        runmat_value::IntValue::U16(2),
        runmat_value::IntValue::U32(3),
        runmat_value::IntValue::U64(u64::MAX),
    ] {
        assert!(run(ispref_builtin(
            Some(runtime.clone()),
            vec![Value::Int(integer)]
        ))
        .is_err());
    }
    assert!(run(ispref_builtin(
        Some(runtime.clone()),
        vec![unregistered_resident_value()]
    ))
    .is_err());
}

#[test]
fn preferences_round_trip() {
    let runtime = preference_context();
    run(setpref_builtin(
        Some(runtime.clone()),
        vec![
            Value::String("runmatTest".to_string()),
            Value::String("answer".to_string()),
            Value::Num(42.0),
        ],
    ))
    .unwrap();
    assert_eq!(
        run(ispref_builtin(
            Some(runtime.clone()),
            vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]
        ))
        .unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run(getpref_builtin(
            Some(runtime.clone()),
            vec![
                Value::String("runmatTest".to_string()),
                Value::String("answer".to_string())
            ]
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
        vec![group.clone(), name.clone(), Value::Num(1.0)],
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
        vec![group.clone(), name.clone(), Value::Num(7.0)],
    ))
    .expect("first isolated store");
    assert_eq!(
        run(ispref_builtin(
            Some(second),
            vec![group.clone(), name.clone()]
        ))
        .expect("second isolated store"),
        Value::Bool(false)
    );
    assert_eq!(
        run(getpref_builtin(Some(first), vec![group, name])).expect("first store retains value"),
        Value::Num(7.0)
    );
}

#[test]
fn nested_resident_default_rejects_before_service_or_provider_effects() {
    let default = Value::Cell(
        CellArray::new(vec![unregistered_resident_value()], 1, 1).expect("nested default"),
    );
    let error = run(getpref_builtin(
        None,
        vec![
            Value::String("nestedDefault".into()),
            Value::String("value".into()),
            default,
        ],
    ))
    .expect_err("default intent rejects before service lookup or gather");
    assert!(error.message().contains("provider-resident default"));
}

#[test]
fn failed_nested_setpref_gather_leaves_the_application_store_unchanged() {
    let runtime = preference_context();
    let group = Value::String("nestedSetprefFailure".into());
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        vec![
            group.clone(),
            Value::String("value".into()),
            Value::Cell(
                CellArray::new(vec![unregistered_resident_value()], 1, 1).expect("nested value"),
            ),
        ],
    ))
    .expect_err("unregistered allocation cannot materialize");
    assert!(error.message().contains("provider") || error.message().contains("allocation"));
    assert_eq!(
        run(ispref_builtin(Some(runtime), vec![group])).expect("no partial setting"),
        Value::Bool(false)
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
            vec![group.clone(), names.clone(), values.clone()]
        ))
        .expect("set both"),
        Value::Num(0.0)
    );
    let queried = run(getpref_builtin(
        Some(runtime.clone()),
        vec![group.clone(), names.clone()],
    ))
    .expect("retrieve both");
    assert_eq!(queried, values);
    let flags = run(ispref_builtin(
        Some(runtime.clone()),
        vec![group.clone(), names],
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
    let flags =
        run(ispref_builtin(Some(runtime.clone()), vec![group, mixed])).expect("mixed query");
    let Value::LogicalArray(flags) = flags else {
        panic!("logical array");
    };
    assert_eq!(flags.shape, vec![1, 2]);
    assert_eq!(flags.data.as_slice(), &[1, 0]);
}

#[test]
fn invalid_multi_value_shape_rejects_before_resident_download_or_store_mutation() {
    let runtime = preference_context();
    let group = Value::String("runmatMultiPreferenceAdmission".into());
    let names = Value::StringArray(
        runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2])
            .expect("names shape"),
    );
    let wrong_shape = Value::Cell(
        CellArray::new_with_shape(
            vec![unregistered_resident_value(), Value::Num(2.0)],
            vec![2, 1],
        )
        .expect("wrong shape"),
    );
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        vec![group.clone(), names.clone(), wrong_shape],
    ))
    .expect_err("shape must reject before any gather");
    assert!(error.message().contains("matching shape"));
    assert_eq!(
        run(ispref_builtin(Some(runtime.clone()), vec![group.clone()])).expect("group query"),
        Value::Bool(false)
    );
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        vec![group, names, Value::Num(1.0)],
    ))
    .expect_err("scalar cannot fill multiple settings");
    assert!(error.message().contains("matching shape"));
}

#[test]
fn preference_output_admission_precedes_storage_and_provider_access() {
    let runtime = preference_context();
    let group = Value::String("runmatPreferenceOutputAdmission".into());
    let _count = crate::output_count::push_output_count(Some(2));
    for call in [
        run(getpref_builtin(
            Some(runtime.clone()),
            vec![group.clone(), Value::String("name".into())],
        )),
        run(ispref_builtin(Some(runtime.clone()), vec![group.clone()])),
        run(setpref_builtin(
            Some(runtime.clone()),
            vec![
                group.clone(),
                Value::String("name".into()),
                unregistered_resident_value(),
            ],
        )),
    ] {
        let error = call.expect_err("too many outputs before effects");
        assert!(error.message().contains("too many output"));
    }
    drop(_count);
    assert_eq!(
        run(ispref_builtin(Some(runtime.clone()), vec![group])).expect("no mutation"),
        Value::Bool(false)
    );
}

#[test]
fn setpref_status_result_is_a_separately_gated_extension() {
    let runtime = preference_context();
    let _count = crate::output_count::push_output_count(Some(1));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let group = Value::String("runmatSetprefStatusAdmission".into());
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        vec![
            group.clone(),
            Value::String("name".into()),
            unregistered_resident_value(),
        ],
    ))
    .expect_err("status extension must reject before gather");
    assert_eq!(
        error.identifier(),
        runmat_builtins::SETPREF_STATUS_EXTENSION.error_identifier
    );
    assert_eq!(
        run(ispref_builtin(Some(runtime.clone()), vec![group])).expect("no mutation"),
        Value::Bool(false)
    );
}

#[test]
fn getpref_descriptor_and_integer_capabilities_cover_documented_forms() {
    assert_eq!(GETPREF_DESCRIPTOR.signatures.len(), 4);
    assert_eq!(GETPREF_INTEGER_CAPABILITIES.len(), 2);
    assert_eq!(GETPREF_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    assert_eq!(
        GETPREF_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::PreserveInput
    );
    assert_eq!(
        GETPREF_INTEGER_CAPABILITIES[0].backend,
        BuiltinIntegerBackendRule::HostOnly
    );
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
            vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference.clone()),
                value.clone(),
            ],
        ))
        .expect("create integer preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(
            Some(runtime.clone()),
            vec![
                Value::String("getprefIntegerScalars".into()),
                Value::String(preference),
            ],
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
            vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference.clone()),
                value.clone(),
            ],
        ))
        .expect("create integer array preference");
        assert_eq!(created, value);
        let stored = run(getpref_builtin(
            Some(runtime.clone()),
            vec![
                Value::String("getprefIntegerArrays".into()),
                Value::String(preference),
            ],
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
        vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            existing.clone(),
        ],
    ))
    .expect("create preference");
    let result = run(getpref_builtin(
        Some(runtime.clone()),
        vec![
            Value::String("getprefExistingWins".into()),
            Value::String("value".into()),
            Value::FunctionHandle("session_function".into()),
        ],
    ))
    .expect("ignored unsupported host default must not narrow existing form");
    assert_eq!(result, existing);
}

#[test]
fn getpref_preserves_arbitrary_object_values_and_zero_argument_structure() {
    let runtime = preference_context();
    let mut object = ObjectInstance::new("PreferencePayload".to_string());
    object.properties.insert(
        "revision".to_string(),
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
    );
    let value = Value::Object(object);
    let created = run(getpref_builtin(
        Some(runtime.clone()),
        vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
            value.clone(),
        ],
    ))
    .expect("create object preference");
    assert_eq!(created, value);
    let stored = run(getpref_builtin(
        Some(runtime.clone()),
        vec![
            Value::String("getprefObjects".into()),
            Value::String("payload".into()),
        ],
    ))
    .expect("read object preference");
    assert_eq!(stored, value);

    let all = run(getpref_builtin(Some(runtime.clone()), Vec::new())).expect("all preferences");
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
        vec![
            Value::String("getprefMultiple".into()),
            names.clone(),
            defaults.clone(),
        ],
    ))
    .expect("create multiple preferences");
    assert_eq!(created, defaults);
    let stored = run(getpref_builtin(
        Some(runtime.clone()),
        vec![Value::String("getprefMultiple".into()), names],
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
        vec![Value::String("getprefMultiple".into()), cellstr],
    ))
    .expect("read cellstr preferences");
    let Value::Cell(stored) = stored else {
        panic!("expected cell result");
    };
    assert_eq!(stored.shape, vec![1, 2]);
}

#[test]
fn getpref_group_query_is_a_separately_gated_extension() {
    let runtime = preference_context();
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run(getpref_builtin(
        Some(runtime.clone()),
        vec![Value::String("group".into())],
    ))
    .expect_err("strict group query");
    assert_eq!(
        error.identifier(),
        GETPREF_GROUP_QUERY_EXTENSION.error_identifier
    );
    drop(strict);

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = run(getpref_builtin(
        Some(runtime.clone()),
        vec![Value::String("group".into())],
    ))
    .expect("extension group query");
    assert!(matches!(result, Value::Struct(_)));
}

#[test]
fn getpref_rejects_resident_controls_and_defaults_before_provider_access() {
    let runtime = preference_context();
    let resident = || unregistered_resident_value();
    let error = run(getpref_builtin(
        Some(runtime.clone()),
        vec![resident(), Value::String("name".into())],
    ))
    .expect_err("resident group");
    assert!(error.message().contains("group must be"));

    let error = run(getpref_builtin(
        Some(runtime.clone()),
        vec![Value::String("getprefResident".into()), resident()],
    ))
    .expect_err("resident preference name");
    assert!(error.message().contains("preference must be"));

    let error = run(getpref_builtin(
        Some(runtime.clone()),
        vec![
            Value::String("getprefResident".into()),
            Value::String("value".into()),
            resident(),
        ],
    ))
    .expect_err("resident default");
    assert!(error.message().contains("provider-resident default"));
}

#[test]
fn getpref_rejects_every_integer_class_in_text_control_positions() {
    let runtime = preference_context();
    for value in [
        Value::Int(runmat_value::IntValue::I8(1)),
        Value::Int(runmat_value::IntValue::I16(1)),
        Value::Int(runmat_value::IntValue::I32(1)),
        Value::Int(runmat_value::IntValue::I64(1)),
        Value::Int(runmat_value::IntValue::U8(1)),
        Value::Int(runmat_value::IntValue::U16(1)),
        Value::Int(runmat_value::IntValue::U32(1)),
        Value::Int(runmat_value::IntValue::U64(1)),
    ] {
        let error = run(getpref_builtin(
            Some(runtime.clone()),
            vec![value.clone(), Value::String("name".into())],
        ))
        .expect_err("integer group");
        assert!(error.message().contains("group must be"));
        let error = run(getpref_builtin(
            Some(runtime.clone()),
            vec![Value::String("getprefIntegerControls".into()), value],
        ))
        .expect_err("integer preference name");
        assert!(error.message().contains("preference must be"));
    }
}
