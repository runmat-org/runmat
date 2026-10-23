use super::*;
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerOutputClassRule, GETPREF_DESCRIPTOR,
    GETPREF_GROUP_QUERY_EXTENSION, GETPREF_INTEGER_CAPABILITIES,
};

#[test]
fn missing_named_preference_preserves_public_diagnostic_and_typed_cause() {
    let error = run(getpref_builtin(
        Some(preference_context()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("app".into()),
            Value::String("absent".into()),
        ]),
    ))
    .expect_err("a missing named preference has no implicit default");
    assert_eq!(
        error.message(),
        "getpref: preference 'app/absent' does not exist"
    );
    assert_eq!(error.identifier(), None);
    assert!(std::error::Error::source(&error).is_some());
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(integer)])
        ))
        .is_err());
    }
    assert!(run(ispref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            unregistered_resident_value()
        ])
    ))
    .is_err());
}

#[test]
fn preference_output_admission_precedes_storage_and_provider_access() {
    let runtime = preference_context();
    let group = Value::String("runmatPreferenceOutputAdmission".into());
    let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for call in [
        run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                group.clone(),
                Value::String("name".into()),
            ]),
        )),
        run(ispref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![group.clone()]),
        )),
        run(setpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                group.clone(),
                Value::String("name".into()),
                unregistered_resident_value(),
            ]),
        )),
    ] {
        let error = call.expect_err("too many outputs before effects");
        assert!(error.message().contains("too many output"));
    }
    drop(_count);
    assert_eq!(
        run(ispref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![group])
        ))
        .expect("no mutation"),
        Value::Bool(false)
    );
}

#[test]
fn setpref_status_result_is_a_separately_gated_extension() {
    let runtime = preference_context();
    let _count = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let group = Value::String("runmatSetprefStatusAdmission".into());
    let error = run(setpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            group.clone(),
            Value::String("name".into()),
            unregistered_resident_value(),
        ]),
    ))
    .expect_err("status extension must reject before gather");
    assert_eq!(
        error.identifier(),
        runmat_builtins::SETPREF_STATUS_EXTENSION.error_identifier
    );
    assert_eq!(
        run(ispref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![group])
        ))
        .expect("no mutation"),
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
fn getpref_group_query_is_a_separately_gated_extension() {
    let runtime = preference_context();
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = run(getpref_builtin(
        Some(runtime.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "group".into(),
        )]),
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "group".into(),
        )]),
    ))
    .expect("extension group query");
    assert!(matches!(result, Value::Struct(_)));
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
            crate::builtins::common::test_support::builtin_arguments(vec![
                value.clone(),
                Value::String("name".into()),
            ]),
        ))
        .expect_err("integer group");
        assert!(error.message().contains("group must be"));
        let error = run(getpref_builtin(
            Some(runtime.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("getprefIntegerControls".into()),
                value,
            ]),
        ))
        .expect_err("integer preference name");
        assert!(error.message().contains("preference must be"));
    }
}
