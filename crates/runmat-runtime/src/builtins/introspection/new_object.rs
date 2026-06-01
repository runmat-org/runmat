use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const NEW_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "obj",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Created class object value.",
}];

const NEW_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for the object to create.",
}];

const NEW_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "obj = new_object(class_name)",
    inputs: &NEW_OBJECT_INPUTS,
    outputs: &NEW_OBJECT_OUTPUT,
}];

pub const NEW_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

#[runtime_builtin(
    name = "new_object",
    category = "introspection",
    summary = "Create a class object instance with declared default properties.",
    keywords = "classdef,object,constructor,new",
    descriptor(crate::builtins::introspection::new_object::NEW_OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::new_object"
)]
pub async fn new_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    crate::create_class_object(class_name).await
}
