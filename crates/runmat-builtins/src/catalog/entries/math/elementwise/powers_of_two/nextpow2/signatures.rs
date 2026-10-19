use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "p",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Smallest exponent p for which 2^p is at least abs(X).",
}];

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric or logical input.",
}];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "p = nextpow2(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    super::NEXTPOW2_ERROR_INVALID_INPUT,
    super::NEXTPOW2_ERROR_INTERNAL,
    super::NEXTPOW2_ERROR_TOO_MANY_OUTPUTS,
];

pub(super) const NEXTPOW2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
