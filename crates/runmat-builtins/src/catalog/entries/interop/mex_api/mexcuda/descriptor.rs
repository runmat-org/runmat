use super::super::contract::BUILD_ERRORS;
use crate::*;

const INPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Native CUDA source files and MEX options as scalar text; RunMat also converts scalar numeric and logical arguments to text.",
}];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "mexcuda(arguments)",
    inputs: INPUTS,
    outputs: &[],
}];
pub const MEXCUDA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: BUILD_ERRORS,
};
