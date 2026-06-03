use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const GET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dependent property backing value.",
}];

const GET_P_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Object receiver.",
}];

const GET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = get.p(obj)",
    inputs: &GET_P_INPUTS,
    outputs: &GET_P_OUTPUT,
}];

const GET_P_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.GET_P.RECEIVER_INVALID",
    identifier: Some("RunMat:GetPReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "get.p: requires object receiver",
}];

pub const GET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &GET_P_ERRORS,
};

#[runtime_builtin(
    name = "get.p",
    category = "introspection",
    summary = "Get dependent property backing value for test/runtime compatibility flows.",
    keywords = "classdef,dependent,property,get",
    descriptor(crate::builtins::introspection::dependent_property::GET_P_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dependent_property"
)]
pub async fn get_p_builtin(obj: Value) -> crate::BuiltinResult<Value> {
    crate::get_p_builtin(obj).await
}

const SET_P_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Updated object receiver.",
}];

const SET_P_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Object receiver.",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Value to assign.",
    },
];

const SET_P_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = set.p(obj, value)",
    inputs: &SET_P_INPUTS,
    outputs: &SET_P_OUTPUT,
}];

const SET_P_ERRORS: [BuiltinErrorDescriptor; 1] = [BuiltinErrorDescriptor {
    code: "RM.SET_P.RECEIVER_INVALID",
    identifier: Some("RunMat:SetPReceiverInvalid"),
    when: "Receiver is not an object value.",
    message: "set.p: requires object receiver",
}];

pub const SET_P_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SET_P_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &SET_P_ERRORS,
};

#[runtime_builtin(
    name = "set.p",
    category = "introspection",
    summary = "Set dependent property backing value for test/runtime compatibility flows.",
    keywords = "classdef,dependent,property,set",
    descriptor(crate::builtins::introspection::dependent_property::SET_P_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::dependent_property"
)]
pub async fn set_p_builtin(obj: Value, val: Value) -> crate::BuiltinResult<Value> {
    crate::set_p_builtin(obj, val).await
}
