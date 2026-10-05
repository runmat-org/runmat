use runmat_builtins::FUNCTIONS_ERROR_HANDLE_UNSUPPORTED;
use runmat_value::{StructValue, Value};

#[derive(Debug, Clone, Copy)]
enum FunctionHandleKind {
    Simple,
    Anonymous,
    Scoped,
}

impl FunctionHandleKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Anonymous => "anonymous",
            Self::Scoped => "scopedfunction",
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

pub(crate) fn dispatch_functions(handle: Value) -> crate::BuiltinResult<Value> {
    match handle {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Ok(metadata_struct(
            name,
            FunctionHandleKind::Simple,
            String::new(),
        )),
        Value::BoundFunctionHandle { name, .. } => Ok(metadata_struct(
            name,
            FunctionHandleKind::Simple,
            String::new(),
        )),
        Value::Closure(closure) => {
            let kind = if closure.function_name.starts_with("@anon")
                || closure.function_name.starts_with("anonymous#")
            {
                FunctionHandleKind::Anonymous
            } else {
                FunctionHandleKind::Scoped
            };
            Ok(metadata_struct(closure.function_name, kind, String::new()))
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
        let value = dispatch_functions(Value::Closure(runmat_value::Closure {
            function_name: "@anon0".into(),
            bound_function: None,
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
}
