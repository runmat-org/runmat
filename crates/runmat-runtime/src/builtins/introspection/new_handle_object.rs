use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const NEW_HANDLE_OBJECT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handle",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "New handle-object instance.",
}];

const NEW_HANDLE_OBJECT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "class_name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for created object.",
}];

const NEW_HANDLE_OBJECT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "handle = new_handle_object(class_name)",
        inputs: &NEW_HANDLE_OBJECT_INPUTS,
        outputs: &NEW_HANDLE_OBJECT_OUTPUT,
    }];

pub const NEW_HANDLE_OBJECT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEW_HANDLE_OBJECT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runtime_builtin(
    name = "new_handle_object",
    category = "introspection",
    summary = "Create a new handle object instance for internal class construction.",
    keywords = "classdef,handle,object,constructor",
    descriptor(crate::builtins::introspection::new_handle_object::NEW_HANDLE_OBJECT_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::new_handle_object"
)]
pub async fn new_handle_object_builtin(class_name: String) -> crate::BuiltinResult<Value> {
    crate::new_handle_object_builtin(class_name).await
}
