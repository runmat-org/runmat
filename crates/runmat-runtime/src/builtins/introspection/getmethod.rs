use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
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

const GETMETHOD_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.GETMETHOD.NAME_INVALID",
        identifier: Some("RunMat:GetMethodNameInvalid"),
        when: "Method name is empty.",
        message: "getmethod: method name must not be empty",
    },
    BuiltinErrorDescriptor {
        code: "RM.GETMETHOD.RECEIVER_UNSUPPORTED",
        identifier: Some("RunMat:GetMethodReceiverUnsupported"),
        when: "Receiver is neither object nor class reference.",
        message: "getmethod: unsupported receiver",
    },
    BuiltinErrorDescriptor {
        code: "RM.GETMETHOD.METHOD_PRIVATE",
        identifier: Some("RunMat:MethodPrivate"),
        when: "Method access is private or protected and the caller scope is not allowed.",
        message: "getmethod: method is not accessible from current scope",
    },
];

pub const GETMETHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GETMETHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GETMETHOD_ERRORS,
};

pub(crate) fn dispatch_getmethod(obj: Value, name: String) -> crate::BuiltinResult<Value> {
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
            &GETMETHOD_ERRORS[2],
            format!("{class_name}.{method_name}"),
        ))
    }

    let method_name = name.trim();
    if method_name.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "getmethod",
            &GETMETHOD_ERRORS[0],
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
            &GETMETHOD_ERRORS[1],
            format!("{other:?}"),
        )),
    }
}

#[runtime_builtin(
    name = "getmethod",
    category = "introspection",
    summary = "Create a method-bound function handle from object/class and method name.",
    keywords = "method,function_handle,classdef,dispatch",
    descriptor(crate::builtins::introspection::getmethod::GETMETHOD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::getmethod"
)]
pub async fn getmethod_builtin(obj: Value, name: String) -> crate::BuiltinResult<Value> {
    dispatch_getmethod(obj, name)
}
