use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const NOTIFY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Always zero on success.",
}];

const NOTIFY_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        name: "varargin",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Event callback arguments.",
    },
];

const NOTIFY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = notify(target, event_name, varargin)",
    inputs: &NOTIFY_INPUTS,
    outputs: &NOTIFY_OUTPUT,
}];

const NOTIFY_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.NOTIFY.TARGET_INVALID",
    identifier: Some("RunMat:NotifyTargetInvalid"),
    when: "Target is not an object or handle object.",
    message: "notify: target must be handle or object",
}];

pub const NOTIFY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NOTIFY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NOTIFY_ERRORS,
};

#[runtime_builtin(
    name = "notify",
    category = "introspection",
    summary = "Dispatch event notifications to registered listeners.",
    keywords = "events,listener,notify,callback,classdef",
    descriptor(crate::builtins::introspection::notify::NOTIFY_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::notify"
)]
pub async fn notify_builtin(
    target: Value,
    event_name: String,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    crate::notify_builtin(target, event_name, rest).await
}
