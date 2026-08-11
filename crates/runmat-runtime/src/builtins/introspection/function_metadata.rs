use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, StructValue, Value,
};

const FUNCTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle metadata struct.",
}];

const FUNCTIONS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle.",
}];

const FUNCTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = functions(fh)",
    inputs: &FUNCTIONS_INPUTS,
    outputs: &FUNCTIONS_OUTPUT,
}];

pub const FUNCTIONS_ERROR_HANDLE_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FUNCTIONS.HANDLE_UNSUPPORTED",
    identifier: Some("RunMat:FunctionsHandleUnsupported"),
    when: "The input is not a supported function handle value.",
    message: "functions: expected a function handle",
};

pub const FUNCTIONS_ERRORS: [BuiltinErrorDescriptor; 1] = [FUNCTIONS_ERROR_HANDLE_UNSUPPORTED];

pub const FUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FUNCTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FUNCTIONS_ERRORS,
};

pub const FUNCTIONS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "functions accepts only function-handle values and returns a metadata struct; integer, numeric, and provider-resident values reject before provider access. Captured values in future workspace metadata are opaque closure state rather than integer signature positions and must remain exact.",
    };

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
    category = "introspection",
    summary = "Return deterministic metadata for a function handle.",
    descriptor(self::FUNCTIONS_DESCRIPTOR),
    integer_audit(self::FUNCTIONS_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::function_metadata"
)]
pub fn functions_builtin_registered(handle: Value) -> crate::BuiltinResult<Value> {
    dispatch_functions(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Struct(info) = value else {
            panic!("expected struct metadata, got {value:?}");
        };
        info.fields
            .get(name)
            .unwrap_or_else(|| panic!("missing field {name}"))
    }

    fn unowned_resident_value() -> Value {
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        })
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
            unowned_resident_value(),
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
        let value = dispatch_functions(Value::Closure(runmat_builtins::Closure {
            function_name: "@anon0".to_string(),
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
