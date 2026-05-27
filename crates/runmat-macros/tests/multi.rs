use runmat_builtins::{
    builtin_functions, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const BIN_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "a",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left integer input.",
    },
    BuiltinParamDescriptor {
        name: "b",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right integer input.",
    },
];
const OUT_INT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Integer output.",
}];
const ADD_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = add(a, b)",
    inputs: &BIN_INPUTS,
    outputs: &OUT_INT,
}];
const SUB_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = sub(a, b)",
    inputs: &BIN_INPUTS,
    outputs: &OUT_INT,
}];
const ADD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};
const SUB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUB_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};

mod inner {
    use super::*;

    #[runtime_builtin(
        name = "add",
        descriptor(crate::ADD_DESCRIPTOR),
        builtin_path = "tests::add"
    )]
    pub fn add(a: i32, b: i32) -> Result<i32, String> {
        Ok(a + b)
    }
}

#[runtime_builtin(
    name = "sub",
    descriptor(crate::SUB_DESCRIPTOR),
    builtin_path = "tests::sub"
)]
pub fn sub(a: i32, b: i32) -> Result<i32, String> {
    Ok(a - b)
}

#[test]
fn registers_multiple_functions() {
    let names: Vec<&str> = builtin_functions().into_iter().map(|b| b.name).collect();
    assert!(names.contains(&"add"));
    assert!(names.contains(&"sub"));
}
