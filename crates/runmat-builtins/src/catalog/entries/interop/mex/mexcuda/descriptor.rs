use super::super::contract::BUILD_ERRORS;
use crate::*;

const INPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Native source files and compatible MEX build options.",
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
