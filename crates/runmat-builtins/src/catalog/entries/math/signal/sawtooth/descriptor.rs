use crate::*;
const SAWTOOTH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sawtooth output sampled at t.",
}];

const SAWTOOTH_SIG_DEFAULT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const SAWTOOTH_SIG_XMAX_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "xmax",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Peak position fraction in [0, 1].",
    },
];

const SAWTOOTH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = sawtooth(t)",
        inputs: &SAWTOOTH_SIG_DEFAULT_INPUTS,
        outputs: &SAWTOOTH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = sawtooth(t, xmax)",
        inputs: &SAWTOOTH_SIG_XMAX_INPUTS,
        outputs: &SAWTOOTH_OUTPUT,
    },
];

pub const SAWTOOTH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.INVALID_INPUT",
    identifier: Some("RunMat:sawtooth:InvalidInput"),
    when: "Primary input is not numeric-real tensor/scalar compatible.",
    message: "sawtooth: expected numeric input",
};

pub const SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:sawtooth:ComplexUnsupported"),
    when: "Primary input includes complex values.",
    message: "sawtooth: input must be real; complex values are not supported",
};

pub const SAWTOOTH_ERROR_XMAX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.XMAX_INVALID",
    identifier: Some("RunMat:sawtooth:XmaxInvalid"),
    when: "xmax argument is not a real numeric scalar.",
    message: "sawtooth: xmax must be a real numeric scalar in [0, 1]",
};

pub const SAWTOOTH_ERROR_XMAX_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.XMAX_RANGE",
    identifier: Some("RunMat:sawtooth:XmaxOutOfRange"),
    when: "xmax argument lies outside [0, 1] or is non-finite.",
    message: "sawtooth: xmax must be a finite scalar in [0, 1]",
};

pub const SAWTOOTH_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.ARG_COUNT",
    identifier: Some("RunMat:sawtooth:ArgCount"),
    when: "More than one optional argument is provided.",
    message: "sawtooth: expected 1 or 2 arguments",
};

pub const SAWTOOTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAWTOOTH.INTERNAL",
    identifier: Some("RunMat:sawtooth:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "sawtooth: internal error",
};

const SAWTOOTH_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SAWTOOTH_ERROR_INVALID_INPUT,
    SAWTOOTH_ERROR_COMPLEX_UNSUPPORTED,
    SAWTOOTH_ERROR_XMAX_INVALID,
    SAWTOOTH_ERROR_XMAX_RANGE,
    SAWTOOTH_ERROR_ARG_COUNT,
    SAWTOOTH_ERROR_INTERNAL,
];

pub const SAWTOOTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAWTOOTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAWTOOTH_ERRORS,
};
