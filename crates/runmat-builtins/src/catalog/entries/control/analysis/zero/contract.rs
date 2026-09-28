use crate::*;

const ZERO_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zeros of the SISO tf model as a column vector.",
}];
const ZERO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO tf model.",
}];
const ZERO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "z = zero(sys)",
    inputs: &ZERO_INPUTS,
    outputs: &ZERO_OUTPUT,
}];
pub const ZERO_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INVALID_MODEL",
        identifier: Some("RunMat:zero:InvalidModel"),
        when: "Input system is not a valid SISO tf object.",
        message: "zero: invalid model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.UNSUPPORTED_MODEL",
        identifier: Some("RunMat:zero:UnsupportedModel"),
        when: "Model form is unsupported.",
        message: "zero: unsupported model",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INVALID_ARGUMENT",
        identifier: Some("RunMat:zero:InvalidArgument"),
        when: "Model metadata or arguments are malformed.",
        message: "zero: invalid argument",
    },
    BuiltinErrorDescriptor {
        code: "RM.ZERO.INTERNAL",
        identifier: Some("RunMat:zero:Internal"),
        when: "Root calculation or output construction failed.",
        message: "zero: internal error",
    },
];
pub const ZERO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ZERO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ZERO_ERRORS,
};
pub const ZERO_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "zero accepts a dynamic-system model; fundamental integer values are invalid model inputs and reject before model payload or provider access.",
};
