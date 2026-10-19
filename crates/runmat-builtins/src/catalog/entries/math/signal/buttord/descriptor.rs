use crate::*;

const BUTTORD_OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minimum Butterworth filter order.",
    },
    BuiltinParamDescriptor {
        name: "Wn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Natural cutoff frequency or frequency pair for butter.",
    },
];

const BUTTORD_INPUTS_DIGITAL: [BuiltinParamDescriptor; 4] = [
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

const BUTTORD_INPUTS_ANALOG: [BuiltinParamDescriptor; 5] = [
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

const BUTTORD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs)",
        inputs: &BUTTORD_INPUTS_DIGITAL,
        outputs: &BUTTORD_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[n, Wn] = buttord(Wp, Ws, Rp, Rs, 's')",
        inputs: &BUTTORD_INPUTS_ANALOG,
        outputs: &BUTTORD_OUTPUTS,
    },
];

pub const BUTTORD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.ARG_COUNT",
    identifier: Some("RunMat:buttord:ArgCount"),
    when: "The argument count is outside supported forms.",
    message: "buttord: expected buttord(Wp, Ws, Rp, Rs [, 's'])",
};

pub const BUTTORD_ERROR_INVALID_FREQUENCY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_FREQUENCY",
    identifier: Some("RunMat:buttord:InvalidFrequency"),
    when: "Passband or stopband edges are invalid.",
    message: "buttord: invalid passband or stopband frequency",
};

pub const BUTTORD_ERROR_INVALID_ATTENUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_ATTENUATION",
    identifier: Some("RunMat:buttord:InvalidAttenuation"),
    when: "Ripple or attenuation values are invalid.",
    message: "buttord: Rp and Rs must be positive finite scalars with Rs > Rp",
};

pub const BUTTORD_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INVALID_OPTION",
    identifier: Some("RunMat:buttord:InvalidOption"),
    when: "The optional selector is not supported.",
    message: "buttord: optional selector must be 's'",
};

pub const BUTTORD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:buttord:TooManyOutputs"),
    when: "More than two output arguments are requested.",
    message: "buttord: too many output arguments",
};

pub const BUTTORD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BUTTORD.INTERNAL",
    identifier: Some("RunMat:buttord:Internal"),
    when: "Output tensor construction fails internally.",
    message: "buttord: internal error",
};

const BUTTORD_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BUTTORD_ERROR_ARG_COUNT,
    BUTTORD_ERROR_INVALID_FREQUENCY,
    BUTTORD_ERROR_INVALID_ATTENUATION,
    BUTTORD_ERROR_INVALID_OPTION,
    BUTTORD_ERROR_TOO_MANY_OUTPUTS,
    BUTTORD_ERROR_INTERNAL,
];

pub const BUTTORD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUTTORD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BUTTORD_ERRORS,
};
