use crate::*;
const TRIPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Triangular pulse samples.",
}];

const TRIPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const TRIPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times relative to the pulse center.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Pulse width.",
    },
];

const TRIPULS_INPUTS_T_W_S: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times relative to the pulse center.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Pulse width.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0"),
        description: "Skew in [-1, 1].",
    },
];

const TRIPULS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T)",
        inputs: &TRIPULS_INPUTS_T,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W)",
        inputs: &TRIPULS_INPUTS_T_W,
        outputs: &TRIPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tripuls(T, W, S)",
        inputs: &TRIPULS_INPUTS_T_W_S,
        outputs: &TRIPULS_OUTPUT,
    },
];

pub const TRIPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INVALID_INPUT",
    identifier: Some("RunMat:tripuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "tripuls: expected real numeric input",
};

pub const TRIPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.WIDTH_INVALID",
    identifier: Some("RunMat:tripuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "tripuls: width must be a positive finite scalar",
};

pub const TRIPULS_ERROR_SKEW_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.SKEW_INVALID",
    identifier: Some("RunMat:tripuls:SkewInvalid"),
    when: "Skew is not a finite scalar in [-1, 1].",
    message: "tripuls: skew must be a finite scalar in [-1, 1]",
};

pub const TRIPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.ARG_COUNT",
    identifier: Some("RunMat:tripuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "tripuls: expected 1, 2, or 3 arguments",
};

pub const TRIPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPULS.INTERNAL",
    identifier: Some("RunMat:tripuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "tripuls: internal error",
};

const TRIPULS_ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRIPULS_ERROR_INVALID_INPUT,
    TRIPULS_ERROR_WIDTH_INVALID,
    TRIPULS_ERROR_SKEW_INVALID,
    TRIPULS_ERROR_ARG_COUNT,
    TRIPULS_ERROR_INTERNAL,
];

pub const TRIPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TRIPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TRIPULS_ERRORS,
};
