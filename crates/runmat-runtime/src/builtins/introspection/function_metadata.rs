use runmat_builtins::FUNCTIONS_ERROR_HANDLE_UNSUPPORTED;
use runmat_value::{StructValue, Value};

#[derive(Debug, Clone, Copy)]
enum FunctionHandleKind {
    Simple,
    Anonymous,
    Scoped,
    Nested,
    Unknown,
}

impl FunctionHandleKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Anonymous => "anonymous",
            Self::Scoped => "scopedfunction",
            Self::Nested => "nested",
            Self::Unknown => "unknown",
        }
    }
}

fn metadata_struct(function: String, kind: FunctionHandleKind, file: String) -> Value {
    let mut info = StructValue::new();
    info.insert("function", Value::String(function));
    info.insert("type", Value::String(kind.as_str().to_string()));
    info.insert("file", Value::String(file));
    Value::Struct(info)
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

fn source_metadata(function: usize) -> Option<(FunctionHandleKind, String)> {
    let info = crate::user_functions::source_function_for(function)?;
    let kind = role_kind(info.role)?;
    let file = crate::source_context::source_info(info.source_id)
        .and_then(|source| source.fullpath_name)
        .map_or_else(String::new, |path| path.to_string());
    Some((kind, file))
}

pub(crate) fn dispatch_functions(handle: Value) -> crate::BuiltinResult<Value> {
    match handle {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Ok(metadata_struct(
            name,
            FunctionHandleKind::Simple,
            String::new(),
        )),
        Value::BoundFunctionHandle {
            name,
            function,
            source_role,
        } => {
            let kind = role_kind(source_role).unwrap_or(FunctionHandleKind::Unknown);
            let file = source_metadata(function).map_or_else(String::new, |(_, file)| file);
            Ok(metadata_struct(name, kind, file))
        }
        Value::Closure(closure) => {
            let source = closure.bound_function.and_then(source_metadata);
            let kind = role_kind(closure.source_role)
                .or_else(|| source.as_ref().map(|(kind, _)| *kind))
                .unwrap_or(FunctionHandleKind::Unknown);
            let file = source.map_or_else(String::new, |(_, file)| file);
            Ok(metadata_struct(closure.function_name, kind, file))
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
        let value = dispatch_functions(Value::FunctionHandle("sin".to_string()))
            .expect("functions succeeds");
        assert_eq!(field(&value, "function"), &Value::String("sin".to_string()));
        assert_eq!(field(&value, "type"), &Value::String("simple".to_string()));
        assert_eq!(field(&value, "file"), &Value::String(String::new()));
    }

    #[test]
    fn functions_reports_anonymous_closure_metadata() {
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![crate::user_functions::SourceFunctionInfo {
                source_id: runmat_types::SourceId(5),
                name: "@anon0".into(),
                function: 1,
                role: runmat_types::FunctionSourceRole::Anonymous,
            }]),
        ));
        let value = dispatch_functions(Value::Closure(runmat_value::Closure {
            function_name: "@anon0".into(),
            bound_function: Some(1),
            source_role: runmat_types::FunctionSourceRole::Unknown,
            captures: Vec::new(),
        }))
        .expect("functions succeeds");
        assert_eq!(
            field(&value, "function"),
            &Value::String("@anon0".to_string())
        );
        assert_eq!(
            field(&value, "type"),
            &Value::String("anonymous".to_string())
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
        )]);
        let _catalog = crate::user_functions::install_source_function_catalog(Some(
            std::sync::Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "metadata_file".into(),
                    function: 1,
                    role: runmat_types::FunctionSourceRole::Primary,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id,
                    name: "child_local".into(),
                    function: 2,
                    role: runmat_types::FunctionSourceRole::Local,
                },
            ]),
        ));
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
    }

    #[test]
    fn unbound_closure_uses_value_owned_role_without_name_heuristics() {
        let anonymous = dispatch_functions(Value::Closure(runmat_value::Closure {
            function_name: "plain_display_name".into(),
            bound_function: None,
            source_role: runmat_types::FunctionSourceRole::Anonymous,
            captures: Vec::new(),
        }))
        .expect("anonymous metadata");
        assert_eq!(
            field(&anonymous, "type"),
            &Value::String("anonymous".into())
        );
        let unknown = dispatch_functions(Value::Closure(runmat_value::Closure {
            function_name: "@anon_display_is_not_authority".into(),
            bound_function: None,
            source_role: runmat_types::FunctionSourceRole::Unknown,
            captures: Vec::new(),
        }))
        .expect("unknown metadata");
        assert_eq!(field(&unknown, "type"), &Value::String("unknown".into()));
        let unknown_bound = dispatch_functions(Value::BoundFunctionHandle {
            name: "display_name".into(),
            function: usize::MAX,
            source_role: runmat_types::FunctionSourceRole::Unknown,
        })
        .expect("unregistered bound handle has unknown source role");
        assert_eq!(
            field(&unknown_bound, "type"),
            &Value::String("unknown".into())
        );
    }
}
