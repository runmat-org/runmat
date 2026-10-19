use crate::*;

const INTERP2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Interpolated values over a 2-D grid.",
}];

const INTERP2_INPUTS_Z_XQ_YQ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
];

const INTERP2_INPUTS_X_Y_Z_XQ_YQ: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid X axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid Y axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
];

const INTERP2_INPUTS_Z_XQ_YQ_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", or not-a-knot \"spline\".",
    },
];

const INTERP2_INPUTS_Z_XQ_YQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", or not-a-knot \"spline\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Extrapolation: spline defaults to extrapolation; other methods default to NaN. Override with \"extrap\" or a scalar fill value.",
    },
];

const INTERP2_INPUTS_X_Y_Z_XQ_YQ_METHOD: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid X axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid Y axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", or not-a-knot \"spline\".",
    },
];

const INTERP2_INPUTS_X_Y_Z_XQ_YQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 7] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid X axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid Y axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", or not-a-knot \"spline\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Extrapolation: spline defaults to extrapolation; other methods default to NaN. Override with \"extrap\" or a scalar fill value.",
    },
];

const INTERP2_INPUTS_Z_XQ_YQ_EXTRAP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP2_INPUTS_X_Y_Z_XQ_YQ_EXTRAP: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid X axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid Y axis vector or mesh.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid sample matrix.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query X coordinates.",
    },
    BuiltinParamDescriptor {
        name: "Yq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query Y coordinates.",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP2_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(Z, Xq, Yq)",
        inputs: &INTERP2_INPUTS_Z_XQ_YQ,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(X, Y, Z, Xq, Yq)",
        inputs: &INTERP2_INPUTS_X_Y_Z_XQ_YQ,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(Z, Xq, Yq, method)",
        inputs: &INTERP2_INPUTS_Z_XQ_YQ_METHOD,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(X, Y, Z, Xq, Yq, method)",
        inputs: &INTERP2_INPUTS_X_Y_Z_XQ_YQ_METHOD,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(Z, Xq, Yq, extrap)",
        inputs: &INTERP2_INPUTS_Z_XQ_YQ_EXTRAP,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(X, Y, Z, Xq, Yq, extrap)",
        inputs: &INTERP2_INPUTS_X_Y_Z_XQ_YQ_EXTRAP,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(Z, Xq, Yq, method, extrap)",
        inputs: &INTERP2_INPUTS_Z_XQ_YQ_METHOD_EXTRAP,
        outputs: &INTERP2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp2(X, Y, Z, Xq, Yq, method, extrap)",
        inputs: &INTERP2_INPUTS_X_Y_Z_XQ_YQ_METHOD_EXTRAP,
        outputs: &INTERP2_OUTPUT,
    },
];

pub const INTERP2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP2.INVALID_ARGUMENT",
    identifier: Some("RunMat:interp2:InvalidArgument"),
    when: "Argument count, method/extrapolation options, or axis/query compatibility is invalid.",
    message: "interp2: invalid argument",
};

pub const INTERP2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP2.INVALID_INPUT",
    identifier: Some("RunMat:interp2:InvalidInput"),
    when: "Grid or query values cannot be converted to numeric interpolation domains.",
    message: "interp2: invalid input",
};

pub const INTERP2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP2.INTERNAL",
    identifier: Some("RunMat:interp2:Internal"),
    when: "Interpolation output construction fails due to internal tensor assembly paths.",
    message: "interp2: internal interpolation failure",
};

const INTERP2_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INTERP2_ERROR_INVALID_ARGUMENT,
    INTERP2_ERROR_INVALID_INPUT,
    INTERP2_ERROR_INTERNAL,
];

pub const INTERP2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERP2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTERP2_ERRORS,
};
