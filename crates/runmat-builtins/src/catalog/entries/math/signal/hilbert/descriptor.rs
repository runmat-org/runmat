use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "z",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Complex analytic signal with real part equal to the input signal.",
}];
const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real-valued signal vector or matrix.",
}];
const INPUT_WITH_LENGTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "x",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real-valued signal vector or matrix.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "FFT length along the first non-singleton dimension.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x)",
        inputs: &INPUT,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "z = hilbert(x, N)",
        inputs: &INPUT_WITH_LENGTH,
        outputs: &OUTPUT,
    },
];

pub const HILBERT_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.ARG_COUNT",
    identifier: Some("RunMat:hilbert:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "hilbert: expected hilbert(X) or hilbert(X, N)",
};
pub const HILBERT_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_LENGTH",
    identifier: Some("RunMat:hilbert:InvalidLength"),
    when: "Length argument N is non-scalar, negative, non-finite, or fractional.",
    message: "hilbert: invalid length argument",
};
pub const HILBERT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INVALID_INPUT",
    identifier: Some("RunMat:hilbert:InvalidInput"),
    when: "Input cannot be converted to a real numeric/logical signal.",
    message: "hilbert: expected real numeric input",
};
pub const HILBERT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HILBERT.INTERNAL",
    identifier: Some("RunMat:hilbert:Internal"),
    when: "FFT execution or tensor shaping fails internally.",
    message: "hilbert: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    HILBERT_ERROR_ARG_COUNT,
    HILBERT_ERROR_INVALID_LENGTH,
    HILBERT_ERROR_INVALID_INPUT,
    HILBERT_ERROR_INTERNAL,
];

pub const HILBERT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
