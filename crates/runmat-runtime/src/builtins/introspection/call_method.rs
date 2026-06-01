use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const CALL_METHOD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Method return value(s).",
}];

const CALL_METHOD_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Method arguments.",
    },
];

const CALL_METHOD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[out] = call_method(base, method, varargin)",
    inputs: &CALL_METHOD_INPUTS,
    outputs: &CALL_METHOD_OUTPUT,
}];

const CALL_METHOD_ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.CALL_METHOD.NAME_INVALID",
        identifier: Some("RunMat:CallMethodNameInvalid"),
        when: "The method name is empty or missing.",
        message: "call_method: method name must not be empty",
    },
    BuiltinErrorDescriptor {
        code: "RM.CALL_METHOD.RECEIVER_INVALID",
        identifier: Some("RunMat:InvalidObjectDispatch"),
        when: "Receiver is not an object or handle object.",
        message: "call_method: requires object receiver",
    },
];

pub const CALL_METHOD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CALL_METHOD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &CALL_METHOD_ERRORS,
};

pub(crate) const CALL_METHOD_ERROR_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CALL_METHOD.NAME_INVALID",
    identifier: Some("RunMat:CallMethodNameInvalid"),
    when: "The method name is empty or missing.",
    message: "call_method: method name must not be empty",
};

pub(crate) const CALL_METHOD_ERROR_RECEIVER_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.CALL_METHOD.RECEIVER_INVALID",
        identifier: Some("RunMat:InvalidObjectDispatch"),
        when: "Receiver is not an object or handle object.",
        message: "call_method: requires object receiver",
    };

pub(crate) async fn dispatch_call_method(
    base: Value,
    method: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let method = method.trim().to_string();
    if method.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "call_method",
            &CALL_METHOD_ERROR_NAME_INVALID,
        ));
    }
    match base {
        receiver @ Value::Object(_) | receiver @ Value::HandleObject(_) => {
            let class_name = crate::object_receiver_class_name(&receiver).ok_or_else(|| {
                crate::runtime_descriptor_error(
                    "call_method",
                    &CALL_METHOD_ERROR_RECEIVER_INVALID,
                )
            })?;
            let mut args = Vec::with_capacity(1 + rest.len());
            args.push(receiver.clone());
            args.extend(rest);
            let requested_outputs = crate::current_requested_outputs();
            match crate::dispatch_object_external_member(
                class_name,
                &method,
                args.clone(),
                requested_outputs,
            )
            .await
            {
                Ok(v) => return Ok(v),
                Err(err) if crate::is_undefined_function_error(&err) => {}
                Err(err) => return Err(err),
            }
            let (identity, fallback_policy) = crate::callable_identity_for_handle_name(&method);
            crate::dispatch_callable_with_policy(identity, fallback_policy, args, requested_outputs)
                .await
        }
        other => Err(crate::runtime_descriptor_error_with_detail(
            "call_method",
            &CALL_METHOD_ERROR_RECEIVER_INVALID,
            format!("unsupported receiver {other:?} for method '{method}'"),
        )),
    }
}

#[runtime_builtin(
    name = "call_method",
    category = "introspection",
    summary = "Dispatch object member calls with MATLAB method semantics.",
    keywords = "classdef,method,dispatch,object",
    descriptor(crate::builtins::introspection::call_method::CALL_METHOD_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::call_method"
)]
pub async fn call_method_builtin(
    base: Value,
    method: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    dispatch_call_method(base, method, rest).await
}
