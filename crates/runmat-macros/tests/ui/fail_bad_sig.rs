use runmat_macros::runtime_builtin;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};

const TEST_ERRORS: [BuiltinErrorDescriptor; 0] = [];
const OOPS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input integer.",
}];
const OOPS_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output integer.",
}];
const OOPS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = oops(x)",
    inputs: &OOPS_INPUTS,
    outputs: &OOPS_OUTPUTS,
}];
const OOPS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OOPS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::HiddenInternal,
    errors: &TEST_ERRORS,
};

#[runtime_builtin(
    name = "oops",
    descriptor(crate::OOPS_DESCRIPTOR),
    builtin_path = "tests::oops"
)]
fn bad(x: i32) -> i32 {
    x
}

fn main() {} 
