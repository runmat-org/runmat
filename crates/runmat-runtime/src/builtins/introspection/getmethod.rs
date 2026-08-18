use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const GETMETHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "fh",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Bound method closure/handle.",
}];

const GETMETHOD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj_or_class",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver or class reference.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
];

const GETMETHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "fh = getmethod(obj_or_class, name)",
    inputs: &GETMETHOD_INPUTS,
    outputs: &GETMETHOD_OUTPUT,
}];

const GETMETHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.NAME_INVALID",
    identifier: Some("RunMat:GetMethodNameInvalid"),
    when: "Method name is empty.",
    message: "getmethod: method name must not be empty",
};

const GETMETHOD_ERROR_RECEIVER_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.RECEIVER_UNSUPPORTED",
    identifier: Some("RunMat:GetMethodReceiverUnsupported"),
    when: "Receiver is neither object nor class reference.",
    message: "getmethod: unsupported receiver",
};

const GETMETHOD_ERROR_METHOD_PRIVATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GETMETHOD.METHOD_PRIVATE",
    identifier: Some("RunMat:MethodPrivate"),
    when: "Resolved method exists but is inaccessible from the current class scope.",
    message: "getmethod: method is not accessible from current scope",
};

const GETMETHOD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    GETMETHOD_ERROR_NAME_INVALID,
    GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
    GETMETHOD_ERROR_METHOD_PRIVATE,
];

pub const GETMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETMETHOD_ERRORS,
};

pub const GETMETHOD_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "getmethod-bound-method-handle",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "getmethod bound method-handle creation is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:GetmethodExtension"),
};

pub const GETMETHOD_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [GETMETHOD_EXTENSION];

pub const GETMETHOD_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "RunMat getmethod accepts an object or class reference and a host text method name only. All eight integer classes and provider-resident numeric values reject without conversion, gather, or provider access; this audit does not imply compatibility with MATLAB methods or ismethod.",
};

pub(crate) fn dispatch_getmethod(obj: Value, name: String) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(&GETMETHOD_EXTENSION, "getmethod")?;

    fn ensure_method_accessible(class_name: &str, method_name: &str) -> crate::BuiltinResult<()> {
        let Some((method, owner)) = runmat_builtins::lookup_method(class_name, method_name) else {
            return Ok(());
        };
        let caller_class = crate::class_access_context();
        let access_allowed = match method.access {
            runmat_builtins::Access::Public => true,
            runmat_builtins::Access::Private => caller_class.as_deref() == Some(owner.as_str()),
            runmat_builtins::Access::Protected => caller_class
                .as_deref()
                .is_some_and(|caller| runmat_builtins::is_class_or_subclass(caller, &owner)),
        };
        if access_allowed {
            return Ok(());
        }
        Err(crate::runtime_descriptor_error_with_detail(
            "getmethod",
            &GETMETHOD_ERROR_METHOD_PRIVATE,
            format!("{}.{}", class_name, method_name),
        ))
    }

    let method_name = name.trim();
    if method_name.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "getmethod",
            &GETMETHOD_ERROR_NAME_INVALID,
        ));
    }
    let caller_scope = crate::class_access_context()
        .map(Value::String)
        .unwrap_or_else(|| Value::String(String::new()));
    match obj {
        Value::Object(o) => {
            ensure_method_accessible(&o.class_name, method_name)?;
            if let Some((resolved, _owner)) =
                runmat_builtins::lookup_method(&o.class_name, method_name)
            {
                return Ok(Value::Closure(runmat_builtins::Closure {
                    function_name: resolved.function_name.clone(),
                    bound_function: crate::user_functions::resolve_semantic_function_by_name(
                        &resolved.function_name,
                    ),
                    captures: vec![Value::Object(o)],
                }));
            }
            Ok(Value::Closure(runmat_builtins::Closure {
                function_name: crate::CALL_BOUND_METHOD_BUILTIN_NAME.to_string(),
                bound_function: None,
                captures: vec![
                    Value::Object(o),
                    Value::String(method_name.to_string()),
                    caller_scope.clone(),
                ],
            }))
        }
        Value::HandleObject(h) => {
            ensure_method_accessible(&h.class_name, method_name)?;
            if let Some((resolved, _owner)) =
                runmat_builtins::lookup_method(&h.class_name, method_name)
            {
                return Ok(Value::Closure(runmat_builtins::Closure {
                    function_name: resolved.function_name.clone(),
                    bound_function: crate::user_functions::resolve_semantic_function_by_name(
                        &resolved.function_name,
                    ),
                    captures: vec![Value::HandleObject(h)],
                }));
            }
            Ok(Value::Closure(runmat_builtins::Closure {
                function_name: crate::CALL_BOUND_METHOD_BUILTIN_NAME.to_string(),
                bound_function: None,
                captures: vec![
                    Value::HandleObject(h),
                    Value::String(method_name.to_string()),
                    caller_scope,
                ],
            }))
        }
        Value::ClassRef(cls) => {
            ensure_method_accessible(&cls, method_name)?;
            crate::builtins::introspection::function_handle_text::dispatch_str2func(Value::String(
                format!("@{cls}.{method_name}"),
            ))
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "getmethod",
            &GETMETHOD_ERROR_RECEIVER_UNSUPPORTED,
            format!("{other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "getmethod",
    category = "introspection",
    summary = "Create a method-bound function handle from object/class and method name.",
    keywords = "method,function_handle,classdef,dispatch",
    extensions(crate::builtins::introspection::getmethod::GETMETHOD_EXTENSIONS),
    integer_audit(crate::builtins::introspection::getmethod::GETMETHOD_INTEGER_AUDIT),
    descriptor(crate::builtins::introspection::getmethod::GETMETHOD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::getmethod"
)]
pub async fn getmethod_builtin(obj: Value, name: Value) -> crate::BuiltinResult<Value> {
    crate::compatibility::ensure_builtin_extension_enabled(&GETMETHOD_EXTENSION, "getmethod")?;
    let name = match name {
        Value::String(name) => name,
        Value::CharArray(chars) if chars.rows == 1 => chars.data.iter().collect(),
        Value::StringArray(array) if array.data.len() == 1 => array.data[0].clone(),
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "getmethod",
                &GETMETHOD_ERROR_NAME_INVALID,
                format!("method name must be host text, got {other:?}"),
            ))
        }
    };
    dispatch_getmethod(obj, name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callable_is_registered_as_a_runmat_only_extension() {
        assert_eq!(GETMETHOD_EXTENSIONS, [GETMETHOD_EXTENSION]);
        assert_eq!(GETMETHOD_EXTENSION.mode, BuiltinExtensionMode::RunMatOnly);
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = futures::executor::block_on(getmethod_builtin(
            Value::ClassRef("Example".into()),
            Value::String("method".into()),
        ))
        .expect_err("strict compatibility rejects RunMat getmethod");
        assert_eq!(error.identifier(), GETMETHOD_EXTENSION.error_identifier);
    }

    #[test]
    fn integer_audit_rejects_all_integer_receivers_and_resident_numeric() {
        assert_eq!(
            GETMETHOD_INTEGER_AUDIT.kind,
            BuiltinIntegerAuditKind::NotApplicable
        );
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        for value in [
            Value::Int(runmat_builtins::IntValue::I8(1)),
            Value::Int(runmat_builtins::IntValue::I16(1)),
            Value::Int(runmat_builtins::IntValue::I32(1)),
            Value::Int(runmat_builtins::IntValue::I64(1)),
            Value::Int(runmat_builtins::IntValue::U8(1)),
            Value::Int(runmat_builtins::IntValue::U16(1)),
            Value::Int(runmat_builtins::IntValue::U32(1)),
            Value::Int(runmat_builtins::IntValue::U64(1)),
        ] {
            let error = futures::executor::block_on(getmethod_builtin(
                value,
                Value::String("method".into()),
            ))
            .expect_err("integer receiver");
            assert_eq!(
                error.identifier(),
                GETMETHOD_ERROR_RECEIVER_UNSUPPORTED.identifier
            );
        }
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 1],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
            descriptor: Default::default(),
        });
        let error = futures::executor::block_on(getmethod_builtin(
            resident,
            Value::String("method".into()),
        ))
        .expect_err("resident receiver");
        assert_eq!(
            error.identifier(),
            GETMETHOD_ERROR_RECEIVER_UNSUPPORTED.identifier
        );

        let name_error = futures::executor::block_on(getmethod_builtin(
            Value::ClassRef("Example".into()),
            Value::Int(runmat_builtins::IntValue::U64(u64::MAX)),
        ))
        .expect_err("integer method name rejects without text conversion");
        assert_eq!(
            name_error.identifier(),
            GETMETHOD_ERROR_NAME_INVALID.identifier
        );
    }
}
