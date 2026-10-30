use runmat_builtins::FUNCTIONS_ERROR_HANDLE_UNSUPPORTED;
use runmat_value::Value;

#[derive(Debug, Clone, Copy)]
enum FunctionHandleKind {
    Simple,
    Anonymous,
    Scoped,
    InvocationScoped,
    Nested,
    Unknown,
}

impl FunctionHandleKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Anonymous => "anonymous",
            Self::Scoped => "scopedfunction",
            Self::InvocationScoped => "scopedcallable",
            Self::Nested => "nested",
            Self::Unknown => "unknown",
        }
    }
}

fn metadata_struct(
    function: runmat_value::Utf8Text,
    kind: FunctionHandleKind,
    file: runmat_value::Utf8Text,
) -> Value {
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    info.insert(
        "function",
        Value::String(runmat_value::StringElement::Text(function)),
    );
    info.insert("type", Value::string(kind.as_str().to_string()));
    info.insert(
        "file",
        Value::String(runmat_value::StringElement::Text(file)),
    );
    Value::Struct(info.finish())
}

fn role_kind(role: runmat_types::FunctionSourceRole) -> Option<FunctionHandleKind> {
    Some(match role {
        runmat_types::FunctionSourceRole::Primary | runmat_types::FunctionSourceRole::Method => {
            FunctionHandleKind::Simple
        }
        runmat_types::FunctionSourceRole::Local => FunctionHandleKind::Scoped,
        runmat_types::FunctionSourceRole::Nested => FunctionHandleKind::Nested,
        runmat_types::FunctionSourceRole::Anonymous => FunctionHandleKind::Anonymous,
        runmat_types::FunctionSourceRole::Unknown | runmat_types::FunctionSourceRole::Synthetic => {
            return None
        }
    })
}

pub(crate) fn dispatch_functions(handle: Value) -> crate::BuiltinResult<Value> {
    match handle {
        Value::ScopedFunctionHandle(_) => Ok(metadata_struct(
            runmat_value::Utf8Text::new("<invocation-scoped callable>".to_owned()),
            FunctionHandleKind::InvocationScoped,
            runmat_value::Utf8Text::default(),
        )),
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Ok(metadata_struct(
            name.into_text_owner(),
            FunctionHandleKind::Simple,
            runmat_value::Utf8Text::default(),
        )),
        Value::BoundFunctionHandle(handle) => {
            let data = handle.into_data();
            let kind = role_kind(data.source_role).unwrap_or(FunctionHandleKind::Unknown);
            let file = data.source_file.unwrap_or_default();
            Ok(metadata_struct(data.name, kind, file))
        }
        Value::Closure(closure) => {
            let data = closure.into_data();
            let kind = role_kind(data.source_role).unwrap_or(FunctionHandleKind::Unknown);
            let file = data.source_file.unwrap_or_default();
            Ok(metadata_struct(data.function_name, kind, file))
        }
        _ => Err(crate::runtime_descriptor_error(
            "functions",
            &FUNCTIONS_ERROR_HANDLE_UNSUPPORTED,
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "functions",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::function_metadata"
)]
pub fn functions_builtin_registered(handle: Value) -> crate::BuiltinResult<Value> {
    dispatch_functions(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{BuiltinIntegerAuditKind, FUNCTIONS_INTEGER_AUDIT};
    use runmat_value::{IntValue, IntegerStorage, Tensor};

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Struct(info) = value else {
            panic!("expected struct metadata, got {value:?}");
        };
        info.fields
            .get(name)
            .unwrap_or_else(|| panic!("missing field {name}"))
    }

    fn unregistered_resident_value() -> Value {
        Value::GpuTensor(
            runmat_accelerate_api::test_support::TestAllocationOwner::new()
                .allocate_zeroed(
                    vec![1, 1],
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .expect("valid unregistered resident fixture")
                .handle,
        )
    }

    #[test]
    fn functions_integer_audit_is_inapplicable() {
        assert_eq!(
            FUNCTIONS_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(FUNCTIONS_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    #[test]
    fn functions_rejects_integer_and_resident_values_before_provider_access() {
        let integers = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for value in integers.into_iter().map(Value::Int).chain([
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I64(vec![i64::MAX]), vec![1, 1])
                    .expect("integer tensor"),
            ),
            unregistered_resident_value(),
        ]) {
            let error = dispatch_functions(value).expect_err("non-handle rejection");
            assert_eq!(
                error.identifier(),
                FUNCTIONS_ERROR_HANDLE_UNSUPPORTED.identifier
            );
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }

    #[test]
    fn functions_reports_named_handle_metadata() {
        let value =
            dispatch_functions(Value::FunctionHandle("sin".into())).expect("functions succeeds");
        assert_eq!(
            field(&value, "function"),
            &Value::String("sin".to_string().into())
        );
        assert_eq!(
            field(&value, "type"),
            &Value::String("simple".to_string().into())
        );
        assert_eq!(field(&value, "file"), &Value::String(String::new().into()));
    }

    #[test]
    fn functions_reports_anonymous_closure_metadata() {
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![crate::user_functions::SourceFunctionInfo {
                source_id: runmat_types::SourceId(5),
                name: "@anon0".into(),
                function: 1,
                role: runmat_types::FunctionSourceRole::Anonymous,
                binding_origin: None,
                source_file: None,
                definition_owners: None,
            }]),
        ))
        .expect("call-port scope");
        let value = dispatch_functions(Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "@anon0".into(),
                binding: Some(runmat_value::BoundFunctionBinding::unavailable(1)),
                source_role: runmat_types::FunctionSourceRole::Anonymous,
                source_file: None,
                captures: Vec::new().into(),
            },
        )))
        .expect("functions succeeds");
        assert_eq!(
            field(&value, "function"),
            &Value::String("@anon0".to_string().into())
        );
        assert_eq!(
            field(&value, "type"),
            &Value::String("anonymous".to_string().into())
        );
    }

    #[test]
    fn bound_function_uses_typed_role_and_registered_fullpath() {
        let source_id = runmat_types::SourceId(6);
        let _source = crate::source_context::replace_source_catalog_with_fullpaths(vec![(
            source_id,
            "metadata_file".into(),
            Some("/tmp/metadata_file.m".into()),
            "function metadata_file; end".into(),
        )])
        .expect("function metadata source catalog admission");
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "metadata_file".into(),
                    function: 1,
                    role: runmat_types::FunctionSourceRole::Primary,
                    binding_origin: None,
                    source_file: Some("/tmp/metadata_file.m".into()),
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "child_local".into(),
                    function: 2,
                    role: runmat_types::FunctionSourceRole::Local,
                    binding_origin: None,
                    source_file: Some("/tmp/metadata_file.m".into()),
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "legacy_role_missing".into(),
                    function: 3,
                    role: runmat_types::FunctionSourceRole::Unknown,
                    binding_origin: None,
                    source_file: Some("/tmp/metadata_file.m".into()),
                    definition_owners: None,
                },
            ]),
        ))
        .expect("call-port scope");
        for (name, function, kind) in [
            ("metadata_file", 1, "simple"),
            ("child_local", 2, "scopedfunction"),
        ] {
            let value = dispatch_functions(crate::user_functions::bound_function_handle(
                name.into(),
                function,
            ))
            .expect("metadata");
            assert_eq!(field(&value, "type"), &Value::String(kind.into()));
            assert_eq!(
                field(&value, "file"),
                &Value::String("/tmp/metadata_file.m".into())
            );
        }
        for (stored_role, expected_type) in [
            (runmat_types::FunctionSourceRole::Local, "scopedfunction"),
            (runmat_types::FunctionSourceRole::Unknown, "unknown"),
        ] {
            let value = dispatch_functions(Value::BoundFunctionHandle(
                runmat_value::BoundFunctionHandle::from_data(
                    runmat_value::BoundFunctionHandleData {
                        name: "legacy_role_missing".into(),
                        function: 3,
                        authority: runmat_value::BoundFunctionAuthority::unavailable(3),
                        source_role: stored_role,
                        source_file: Some("/tmp/metadata_file.m".into()),
                    },
                ),
            ))
            .expect("file provenance independent of role");
            assert_eq!(field(&value, "type"), &Value::String(expected_type.into()));
            assert_eq!(
                field(&value, "file"),
                &Value::String("/tmp/metadata_file.m".into())
            );
        }
    }

    #[test]
    fn unbound_closure_uses_value_owned_role_without_name_heuristics() {
        let anonymous = dispatch_functions(Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "plain_display_name".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Anonymous,
                source_file: None,
                captures: Vec::new().into(),
            },
        )))
        .expect("anonymous metadata");
        assert_eq!(
            field(&anonymous, "type"),
            &Value::String("anonymous".into())
        );
        let unknown = dispatch_functions(Value::Closure(runmat_value::Closure::from_data(
            runmat_value::ClosureData {
                lexical_environment: None,
                function_name: "@anon_display_is_not_authority".into(),
                binding: None,
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
                captures: Vec::new().into(),
            },
        )))
        .expect("unknown metadata");
        assert_eq!(field(&unknown, "type"), &Value::String("unknown".into()));
        let unknown_bound = dispatch_functions(Value::BoundFunctionHandle(
            runmat_value::BoundFunctionHandle::from_data(runmat_value::BoundFunctionHandleData {
                name: "display_name".into(),
                function: usize::MAX,
                authority: runmat_value::BoundFunctionAuthority::unavailable(usize::MAX),
                source_role: runmat_types::FunctionSourceRole::Unknown,
                source_file: None,
            }),
        ))
        .expect("unregistered bound handle has unknown source role");
        assert_eq!(
            field(&unknown_bound, "type"),
            &Value::String("unknown".into())
        );
    }

    #[test]
    fn transferred_handle_does_not_rebind_file_from_reused_numeric_function_id() {
        let handle = {
            let source_id = runmat_types::SourceId(41);
            let _source = crate::source_context::replace_source_catalog_with_fullpaths(vec![(
                source_id,
                "origin".into(),
                Some("/origin/origin.m".into()),
                "function origin; end".into(),
            )])
            .expect("function metadata source catalog admission");
            let _catalog = crate::user_functions::install_source_function_catalog(Some(
                std::sync::Arc::new(vec![crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "origin".into(),
                    function: 9,
                    role: runmat_types::FunctionSourceRole::Primary,
                    binding_origin: None,
                    source_file: Some("/origin/origin.m".into()),
                    definition_owners: None,
                }]),
            ))
            .expect("call-port scope");
            crate::user_functions::bound_function_handle("origin".into(), 9)
        };
        let source_id = runmat_types::SourceId(42);
        let _source = crate::source_context::replace_source_catalog_with_fullpaths(vec![(
            source_id,
            "unrelated".into(),
            Some("/destination/unrelated.m".into()),
            "function unrelated; end".into(),
        )])
        .expect("function metadata source catalog admission");
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![crate::user_functions::SourceFunctionInfo {
                source_id,
                name: "unrelated".into(),
                function: 9,
                role: runmat_types::FunctionSourceRole::Local,
                binding_origin: None,
                source_file: Some("/destination/unrelated.m".into()),
                definition_owners: None,
            }]),
        ))
        .expect("call-port scope");
        let info = dispatch_functions(handle).expect("metadata reads original binding");
        assert_eq!(field(&info, "type"), &Value::String("simple".into()));
        assert_eq!(
            field(&info, "file"),
            &Value::String("/origin/origin.m".into())
        );
    }
}
