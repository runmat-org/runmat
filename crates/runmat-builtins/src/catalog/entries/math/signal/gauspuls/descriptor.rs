use crate::*;

const GAUSPULS_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "YI",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "In-phase Gaussian pulse samples.",
    },
    BuiltinParamDescriptor {
        name: "YQ",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Quadrature Gaussian pulse samples.",
    },
    BuiltinParamDescriptor {
        name: "YE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Gaussian pulse envelope samples.",
    },
];

const GAUSPULS_CUTOFF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "TC",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Cutoff time where the envelope reaches TPE dB.",
}];

const GAUSPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const GAUSPULS_INPUTS_T_FC_BW_BWR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "FC",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1000"),
        description: "Carrier frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "BW",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.5"),
        description: "Fractional bandwidth.",
    },
    BuiltinParamDescriptor {
        name: "BWR",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-6"),
        description: "Bandwidth reference level in dB.",
    },
];

const GAUSPULS_INPUTS_CUTOFF: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "`\"cutoff\"` requests cutoff time.",
    },
    BuiltinParamDescriptor {
        name: "FC",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1000"),
        description: "Carrier frequency in Hz.",
    },
    BuiltinParamDescriptor {
        name: "BW",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.5"),
        description: "Fractional bandwidth.",
    },
    BuiltinParamDescriptor {
        name: "BWR",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-6"),
        description: "Bandwidth reference level in dB.",
    },
    BuiltinParamDescriptor {
        name: "TPE",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("-60"),
        description: "Trailing pulse envelope level in dB.",
    },
];

const GAUSPULS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = gauspuls(T)",
        inputs: &GAUSPULS_INPUTS_T,
        outputs: &GAUSPULS_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[YI, YQ, YE] = gauspuls(T, FC, BW, BWR)",
        inputs: &GAUSPULS_INPUTS_T_FC_BW_BWR,
        outputs: &GAUSPULS_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "TC = gauspuls(\"cutoff\", FC, BW, BWR, TPE)",
        inputs: &GAUSPULS_INPUTS_CUTOFF,
        outputs: &GAUSPULS_CUTOFF_OUTPUT,
    },
];

pub const GAUSPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INVALID_INPUT",
    identifier: Some("RunMat:gauspuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "gauspuls: expected real numeric input",
};

pub const GAUSPULS_ERROR_INVALID_PARAMETER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INVALID_PARAMETER",
    identifier: Some("RunMat:gauspuls:InvalidParameter"),
    when: "Frequency, bandwidth, reference level, or cutoff level is malformed.",
    message: "gauspuls: invalid parameter",
};

pub const GAUSPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.ARG_COUNT",
    identifier: Some("RunMat:gauspuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message:
        "gauspuls: expected gauspuls(T, FC, BW, BWR) or gauspuls(\"cutoff\", FC, BW, BWR, TPE)",
};

pub const GAUSPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GAUSPULS.INTERNAL",
    identifier: Some("RunMat:gauspuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "gauspuls: internal error",
};

const GAUSPULS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    GAUSPULS_ERROR_INVALID_INPUT,
    GAUSPULS_ERROR_INVALID_PARAMETER,
    GAUSPULS_ERROR_ARG_COUNT,
    GAUSPULS_ERROR_INTERNAL,
];

pub const GAUSPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GAUSPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GAUSPULS_ERRORS,
};
