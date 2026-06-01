use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const CLASSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ref",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class reference value.",
}];

const CLASSREF_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name.",
}];

const CLASSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "ref = classref(class_name)",
    inputs: &CLASSREF_INPUTS,
    outputs: &CLASSREF_OUTPUT,
}];

pub const CLASSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runtime_builtin(
    name = "classref",
    category = "introspection",
    summary = "Create a class reference token for static dispatch flows.",
    keywords = "classdef,classref,meta.class",
    descriptor(crate::builtins::introspection::classref::CLASSREF_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::classref"
)]
pub async fn classref_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    crate::classref_builtin(class_name).await
}
