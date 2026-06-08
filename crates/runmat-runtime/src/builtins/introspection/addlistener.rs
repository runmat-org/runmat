use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const ADDLISTENER_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "listener",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Created listener handle.",
}];

const ADDLISTENER_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "target",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target object or handle.",
    },
    BuiltinParamDescriptor {
        name: "event_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Event name.",
    },
    BuiltinParamDescriptor {
        name: "callback",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Callback handle or text handle.",
    },
];

const ADDLISTENER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "listener = addlistener(target, event_name, callback)",
    inputs: &ADDLISTENER_INPUTS,
    outputs: &ADDLISTENER_OUTPUT,
}];

const ADDLISTENER_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.ADDLISTENER.TARGET_INVALID",
    identifier: Some("RunMat:AddListenerTargetInvalid"),
    when: "Target is not an object or handle object.",
    message: "addlistener: target must be handle or object",
}];

pub const ADDLISTENER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADDLISTENER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDLISTENER_ERRORS,
};

#[runtime_builtin(
    name = "addlistener",
    category = "introspection",
    summary = "Register callback listeners on object events.",
    keywords = "events,listener,notify,handle,classdef",
    descriptor(crate::builtins::introspection::addlistener::ADDLISTENER_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::addlistener"
)]
pub async fn addlistener_builtin(
    target: Value,
    event_name: String,
    callback: Value,
) -> crate::BuiltinResult<Value> {
    crate::addlistener_builtin(target, event_name, callback).await
}
