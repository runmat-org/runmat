use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerAuditDescriptor,
    BuiltinIntegerAuditKind, BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor,
    BuiltinParamType, BuiltinSignatureDescriptor, Value,
};

const STR2FUNC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

const STR2FUNC_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle text.",
}];

const STR2FUNC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = str2func(name)",
    inputs: &STR2FUNC_INPUTS,
    outputs: &STR2FUNC_OUTPUT,
}];

pub const STR2FUNC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STR2FUNC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STR2FUNC_ERRORS,
};

pub const STR2FUNC_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "str2func accepts scalar function-name text only. Integer, numeric, and provider-resident values reject before provider access, and the builtin has no integer output surface.",
    };

const STR2FUNC_ERROR_NAME_SHAPE_INVALID: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_SHAPE_INVALID",
        identifier: Some("RunMat:Str2FuncNameShapeInvalid"),
        when: "Input text value is not scalar row text.",
        message: "str2func: function name text input must be scalar row text",
    };

const STR2FUNC_ERROR_NAME_TYPE_INVALID: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_TYPE_INVALID",
        identifier: Some("RunMat:Str2FuncNameTypeInvalid"),
        when: "Input is not string or char text.",
        message: "str2func: expected string/char function name",
    };

const STR2FUNC_ERROR_NAME_INVALID: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::BuiltinErrorDescriptor {
        code: "RM.STR2FUNC.NAME_INVALID",
        identifier: Some("RunMat:Str2FuncNameInvalid"),
        when: "Parsed function name is empty.",
        message: "str2func: function name must not be empty",
    };

pub(crate) const STR2FUNC_ERRORS: [runmat_builtins::BuiltinErrorDescriptor; 3] = [
    STR2FUNC_ERROR_NAME_SHAPE_INVALID,
    STR2FUNC_ERROR_NAME_TYPE_INVALID,
    STR2FUNC_ERROR_NAME_INVALID,
];

pub(crate) fn dispatch_str2func(value: Value) -> crate::BuiltinResult<Value> {
    fn normalize_handle_name(text: &str) -> Option<String> {
        let trimmed = text.trim();
        let name = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
        (!name.is_empty()).then(|| name.to_string())
    }

    let name = match value {
        Value::String(text) => normalize_handle_name(&text),
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_handle_name(&sa.data[0]),
        Value::StringArray(_) => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "string array must be scalar",
            ))
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_handle_name(&text)
        }
        Value::CharArray(_) => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_SHAPE_INVALID,
                "char array must be a row vector",
            ))
        }
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "str2func",
                &STR2FUNC_ERROR_NAME_TYPE_INVALID,
                format!("got {other:?}"),
            ))
        }
    }
    .ok_or_else(|| crate::runtime_descriptor_error("str2func", &STR2FUNC_ERROR_NAME_INVALID))?;

    if let Some(function) = crate::user_functions::resolve_semantic_function_by_name(&name) {
        Ok(Value::BoundFunctionHandle { name, function })
    } else if crate::is_well_formed_qualified_name(&name) {
        Ok(Value::ExternalFunctionHandle(name))
    } else {
        Ok(Value::FunctionHandle(name))
    }
}

#[runmat_macros::runtime_builtin(
    name = "str2func",
    descriptor(self::STR2FUNC_DESCRIPTOR),
    integer_audit(self::STR2FUNC_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::function_handle_text"
)]
pub fn str2func_builtin_registered(value: Value) -> crate::BuiltinResult<Value> {
    dispatch_str2func(value)
}

const FUNC2STR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function name string.",
}];

const FUNC2STR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Function handle value.",
}];

const FUNC2STR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = func2str(fh)",
    inputs: &FUNC2STR_INPUTS,
    outputs: &FUNC2STR_OUTPUT,
}];

pub const FUNC2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FUNC2STR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FUNC2STR_ERRORS,
};

pub const FUNC2STR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "func2str accepts only function-handle values and returns handle text; integer, numeric, and provider-resident values reject before provider access.",
    };

const FUNC2STR_ERROR_HANDLE_TYPE_INVALID: runmat_builtins::BuiltinErrorDescriptor =
    runmat_builtins::BuiltinErrorDescriptor {
        code: "RM.FUNC2STR.HANDLE_TYPE_INVALID",
        identifier: Some("RunMat:Func2StrHandleTypeInvalid"),
        when: "Input is not a supported function handle value.",
        message: "func2str: expected function handle",
    };

pub(crate) const FUNC2STR_ERRORS: [runmat_builtins::BuiltinErrorDescriptor; 1] =
    [FUNC2STR_ERROR_HANDLE_TYPE_INVALID];

pub(crate) fn dispatch_func2str(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name)
        | Value::BoundFunctionHandle { name, .. } => Ok(Value::String(name)),
        Value::Closure(closure) => Ok(Value::String(closure.function_name)),
        other => Err(crate::runtime_descriptor_error_with_detail(
            "func2str",
            &FUNC2STR_ERROR_HANDLE_TYPE_INVALID,
            format!("got {other:?}"),
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "func2str",
    descriptor(self::FUNC2STR_DESCRIPTOR),
    integer_audit(self::FUNC2STR_INTEGER_AUDIT),
    builtin_path = "crate::builtins::introspection::function_handle_text"
)]
pub fn func2str_builtin_registered(value: Value) -> crate::BuiltinResult<Value> {
    dispatch_func2str(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::{IntValue, IntegerStorage, Tensor};

    fn unowned_resident_value() -> Value {
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        })
    }

    #[test]
    fn str2func_integer_audit_is_inapplicable() {
        assert_eq!(
            STR2FUNC_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(STR2FUNC_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    #[test]
    fn str2func_rejects_integer_and_resident_values_before_provider_access() {
        for value in [
            Value::Int(IntValue::U64(u64::MAX)),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("integer tensor"),
            ),
            unowned_resident_value(),
        ] {
            let error = dispatch_str2func(value).expect_err("non-text must reject");
            assert_eq!(error.identifier(), Some("RunMat:Str2FuncNameTypeInvalid"));
        }
    }

    #[test]
    fn func2str_integer_audit_is_inapplicable() {
        assert_eq!(
            FUNC2STR_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        assert!(FUNC2STR_INTEGER_AUDIT.canonical_builtin.is_none());
    }

    #[test]
    fn func2str_rejects_integer_and_resident_values_before_provider_access() {
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
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                    .expect("integer tensor"),
            ),
            unowned_resident_value(),
        ]) {
            let error = dispatch_func2str(value).expect_err("non-handle rejection");
            assert_eq!(
                error.identifier(),
                FUNC2STR_ERROR_HANDLE_TYPE_INVALID.identifier
            );
            assert!(!error.message().to_ascii_lowercase().contains("provider"));
        }
    }
}
