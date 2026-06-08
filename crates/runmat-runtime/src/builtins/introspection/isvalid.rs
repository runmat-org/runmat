use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const ISVALID_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when handle/listener is valid.",
}];

const ISVALID_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Value to inspect.",
}];

const ISVALID_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvalid(value)",
    inputs: &ISVALID_INPUTS,
    outputs: &ISVALID_OUTPUT,
}];

pub const ISVALID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISVALID_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runtime_builtin(
    name = "isvalid",
    category = "introspection",
    summary = "Return true for valid handles and listeners.",
    keywords = "handle,listener,validity,classdef",
    descriptor(crate::builtins::introspection::isvalid::ISVALID_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::isvalid"
)]
pub async fn isvalid_builtin(v: Value) -> crate::BuiltinResult<Value> {
    crate::isvalid_builtin(v).await
}
