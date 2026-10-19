use crate::*;

const CHEB2ORD_OUTPUT_N: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Minimum Chebyshev Type II filter order.",
}];

const CHEB2ORD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum Chebyshev Type II filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Natural cutoff frequency or frequency pair for Chebyshev Type II design.",
    },
];

const CHEB2ORD_INPUTS_DIGITAL: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Passband edge or two-element passband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stopband edge or two-element stopband edge vector.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
];

const CHEB2ORD_INPUTS_ANALOG: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Wp",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog passband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Ws",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog stopband edge or edge pair in rad/s.",
    },
    BuiltinParamDescriptor {
        name: "Rp",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Maximum passband ripple in dB.",
    },
    BuiltinParamDescriptor {
        name: "Rs",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum stopband attenuation in dB.",
    },
    BuiltinParamDescriptor {
        name: "s",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Analog-design selector.",
    },
];

const CHEB2ORD_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "n = cheb2ord(Wp, Ws, Rp, Rs)",
        inputs: &CHEB2ORD_INPUTS_DIGITAL,
        outputs: &CHEB2ORD_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs)",
        inputs: &CHEB2ORD_INPUTS_DIGITAL,
        outputs: &CHEB2ORD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "n = cheb2ord(Wp, Ws, Rp, Rs, 's')",
        inputs: &CHEB2ORD_INPUTS_ANALOG,
        outputs: &CHEB2ORD_OUTPUT_N,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = cheb2ord(Wp, Ws, Rp, Rs, 's')",
        inputs: &CHEB2ORD_INPUTS_ANALOG,
        outputs: &CHEB2ORD_OUTPUTS,
    },
];

pub const CHEB2ORD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.ARG_COUNT",
    identifier: Some("RunMat:cheb2ord:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "cheb2ord: expected cheb2ord(Wp, Ws, Rp, Rs [, 's'])",
};

pub const CHEB2ORD_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_FREQUENCY",
    identifier: Some("RunMat:cheb2ord:InvalidFrequency"),
    when: "Passband or stopband edges are invalid.",
    message: "cheb2ord: invalid passband or stopband frequency",
};

pub const CHEB2ORD_ERROR_INVALID_ATTENUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_ATTENUATION",
    identifier: Some("RunMat:cheb2ord:InvalidAttenuation"),
    when: "Ripple or attenuation values are invalid.",
    message: "cheb2ord: Rp and Rs must be positive finite scalars with Rs > Rp",
};

pub const CHEB2ORD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INVALID_OPTION",
    identifier: Some("RunMat:cheb2ord:InvalidOption"),
    when: "The optional selector is not supported.",
    message: "cheb2ord: optional selector must be 's'",
};

pub const CHEB2ORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:cheb2ord:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "cheb2ord: too many output arguments",
};

pub const CHEB2ORD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHEB2ORD.INTERNAL",
    identifier: Some("RunMat:cheb2ord:Internal"),
    when: "Resident input materialization or output tensor construction fails operationally.",
    message: "cheb2ord: internal error",
};

const CHEB2ORD_ERRORS: [BuiltinErrorDescriptor; 6] = [
    CHEB2ORD_ERROR_ARG_COUNT,
    CHEB2ORD_ERROR_INVALID_FREQUENCY,
    CHEB2ORD_ERROR_INVALID_ATTENUATION,
    CHEB2ORD_ERROR_INVALID_OPTION,
    CHEB2ORD_ERROR_TOO_MANY_OUTPUTS,
    CHEB2ORD_ERROR_INTERNAL,
];

pub const CHEB2ORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHEB2ORD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHEB2ORD_ERRORS,
};
