use crate::*;
const RECTPULS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rectangular pulse samples.",
}];

const RECTPULS_INPUTS_T: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times relative to the pulse center.",
}];

const RECTPULS_INPUTS_T_W: [BuiltinParamDescriptor; 2] = [
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

const RECTPULS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T)",
        inputs: &RECTPULS_INPUTS_T,
        outputs: &RECTPULS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = rectpuls(T, W)",
        inputs: &RECTPULS_INPUTS_T_W,
        outputs: &RECTPULS_OUTPUT,
    },
];

pub const RECTPULS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INVALID_INPUT",
    identifier: Some("RunMat:rectpuls:InvalidInput"),
    when: "Input times cannot be interpreted as real numeric samples.",
    message: "rectpuls: expected real numeric input",
};

pub const RECTPULS_ERROR_WIDTH_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.WIDTH_INVALID",
    identifier: Some("RunMat:rectpuls:WidthInvalid"),
    when: "Width is not a positive finite scalar.",
    message: "rectpuls: width must be a positive finite scalar",
};

pub const RECTPULS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.ARG_COUNT",
    identifier: Some("RunMat:rectpuls:ArgCount"),
    when: "Too many input arguments are provided.",
    message: "rectpuls: expected 1 or 2 arguments",
};

pub const RECTPULS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RECTPULS.INTERNAL",
    identifier: Some("RunMat:rectpuls:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "rectpuls: internal error",
};

const RECTPULS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    RECTPULS_ERROR_INVALID_INPUT,
    RECTPULS_ERROR_WIDTH_INVALID,
    RECTPULS_ERROR_ARG_COUNT,
    RECTPULS_ERROR_INTERNAL,
];

pub const RECTPULS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RECTPULS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RECTPULS_ERRORS,
};
