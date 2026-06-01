use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const CALL_BOUND_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Method return value(s).",
}];

const CALL_BOUND_METHOD_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "base",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Method name.",
    },
    BuiltinParamDescriptor {
        name: "scope",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Creator class scope for access checks.",
    },
    BuiltinParamDescriptor {
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method arguments.",
    },
];

const CALL_BOUND_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out] = __runmat_call_bound_method__(base, method, scope, varargin)",
    inputs: &CALL_BOUND_METHOD_INPUTS,
    outputs: &CALL_BOUND_METHOD_OUTPUT,
}];

const CALL_BOUND_METHOD_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.CALL_BOUND_METHOD.SCOPE_INVALID",
    identifier: Some("RunMat:CallMethodScopeInvalid"),
    when: "Scope argument is not a string scalar.",
    message: "__runmat_call_bound_method__: invalid scope argument",
}];

pub const CALL_BOUND_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_BOUND_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &CALL_BOUND_METHOD_ERRORS,
};

#[runtime_builtin(
    name = "__runmat_call_bound_method__",
    category = "introspection",
    summary = "Internal bound-method closure invocation with creator class scope.",
    keywords = "classdef,method,dispatch,closure,internal",
    descriptor(crate::builtins::introspection::call_bound_method::CALL_BOUND_METHOD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::call_bound_method"
)]
pub async fn call_bound_method_builtin(
    base: Value,
    method: String,
    scope: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let scope_class = match scope {
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        other => {
            return Err(crate::runtime_descriptor_error_with_detail(
                "__runmat_call_bound_method__",
                &CALL_BOUND_METHOD_ERRORS[0],
                format!("expected scope string, got {other:?}"),
            ))
        }
    };
    let _scope_guard = scope_class.map(|class_name| crate::push_class_access_context(Some(class_name)));
    let class_name = crate::object_receiver_class_name(&base).ok_or_else(|| {
        crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &crate::builtins::introspection::call_method::CALL_METHOD_ERROR_RECEIVER_INVALID,
        )
    })?;
    let method = method.trim().to_string();
    if method.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "__runmat_call_bound_method__",
            &crate::builtins::introspection::call_method::CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    let mut args = Vec::with_capacity(1 + rest.len());
    args.push(base);
    args.extend(rest);
    let requested_outputs = crate::current_requested_outputs();
    if let Some((_resolved, owner)) = runmat_builtins::lookup_method(&class_name, &method) {
        return crate::dispatch_object_external_member(owner, &method, args, requested_outputs).await;
    }
    crate::dispatch_object_external_member(class_name, &method, args, requested_outputs).await
}
