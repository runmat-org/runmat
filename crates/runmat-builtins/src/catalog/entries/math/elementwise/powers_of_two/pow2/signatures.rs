use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Power-of-two or binary-scaled result.",
}];
const UNARY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Exponent input for 2.^E.",
}];
const BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "F",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Significand input.",
    },
    BuiltinParamDescriptor {
        name: "E",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Binary exponent input.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pow2(X)",
        inputs: &UNARY_INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pow2(F, E)",
        inputs: &BINARY_INPUTS,
        outputs: &OUTPUTS,
    },
];
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    super::POW2_ERROR_INVALID_ARGUMENT,
    super::POW2_ERROR_INVALID_INPUT,
    super::POW2_ERROR_SIZE_MISMATCH,
    super::POW2_ERROR_INTERNAL,
    super::POW2_ERROR_TOO_MANY_OUTPUTS,
];

pub(super) const POW2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
