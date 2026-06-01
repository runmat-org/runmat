use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

const REGISTER_TEST_CLASSES_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Registration status value.",
}];

const REGISTER_TEST_CLASSES_INPUTS: [BuiltinParamDescriptor; 0] = [];

const REGISTER_TEST_CLASSES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "status = __register_test_classes()",
        inputs: &REGISTER_TEST_CLASSES_INPUTS,
        outputs: &REGISTER_TEST_CLASSES_OUTPUT,
    }];

pub const REGISTER_TEST_CLASSES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REGISTER_TEST_CLASSES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &[],
};

#[runtime_builtin(
    name = "__register_test_classes",
    category = "introspection",
    summary = "Register internal test classes used by runtime/unit contract tests.",
    keywords = "test,classdef,registration,internal",
    descriptor(crate::builtins::introspection::test_classes::REGISTER_TEST_CLASSES_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::test_classes"
)]
pub async fn register_test_classes_builtin() -> crate::BuiltinResult<Value> {
    crate::register_test_classes_builtin().await
}
