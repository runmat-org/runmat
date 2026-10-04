use super::super::contract::EXTENSION_ERRORS;
use crate::*;

const OUTPUTS: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "extension",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform-specific MEX filename extension without a leading dot.",
}];
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "extension = mexext",
    inputs: &[],
    outputs: OUTPUTS,
}];
pub const MEXEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: EXTENSION_ERRORS,
};
