use crate::*;

const INTERP1_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Interpolated values at query points.",
}];

const INTERP1_INPUTS_Y_XQ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at X = 1:n along Y's first non-singleton sample dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const INTERP1_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const INTERP1_INPUTS_Y_XQ_METHOD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at X = 1:n along Y's first non-singleton sample dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description:
            "Interpolation method: \"linear\", \"nearest\", \"spline\", \"pchip\", \"cubic\", or \"makima\".",
    },
];

const INTERP1_INPUTS_Y_XQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at X = 1:n along Y's first non-singleton sample dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description:
            "Interpolation method: \"linear\", \"nearest\", \"spline\", \"pchip\", \"cubic\", or \"makima\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Extrapolation mode: \"extrap\" or scalar fill value; when omitted, the default depends on the method.",
    },
];

const INTERP1_INPUTS_X_Y_XQ_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description:
            "Interpolation method: \"linear\", \"nearest\", \"spline\", \"pchip\", \"cubic\", or \"makima\".",
    },
];

const INTERP1_INPUTS_X_Y_XQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description:
            "Interpolation method: \"linear\", \"nearest\", \"spline\", \"pchip\", \"cubic\", or \"makima\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Extrapolation mode: \"extrap\" or scalar fill value; when omitted, the default depends on the method.",
    },
];

const INTERP1_INPUTS_Y_XQ_EXTRAP: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at X = 1:n along Y's first non-singleton sample dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP1_INPUTS_X_Y_XQ_EXTRAP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP1_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq)",
        inputs: &INTERP1_INPUTS_Y_XQ,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq)",
        inputs: &INTERP1_INPUTS_X_Y_XQ,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, method)",
        inputs: &INTERP1_INPUTS_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, method)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, extrap)",
        inputs: &INTERP1_INPUTS_Y_XQ_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, extrap)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, method, extrap)",
        inputs: &INTERP1_INPUTS_Y_XQ_METHOD_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, method, extrap)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_METHOD_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
];

pub const INTERP1_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INVALID_ARGUMENT",
    identifier: Some("RunMat:interp1:InvalidArgument"),
    when: "Argument count, method/extrapolation options, or shape constraints are invalid.",
    message: "interp1: invalid argument",
};

pub const INTERP1_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INVALID_INPUT",
    identifier: Some("RunMat:interp1:InvalidInput"),
    when: "Sample or query values cannot be converted to numeric interpolation domains.",
    message: "interp1: invalid input",
};

pub const INTERP1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INTERNAL",
    identifier: Some("RunMat:interp1:Internal"),
    when: "Interpolation evaluation fails due to internal tensor construction or solver paths.",
    message: "interp1: internal interpolation failure",
};

const INTERP1_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INTERP1_ERROR_INVALID_ARGUMENT,
    INTERP1_ERROR_INVALID_INPUT,
    INTERP1_ERROR_INTERNAL,
];

pub const INTERP1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERP1_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTERP1_ERRORS,
};
