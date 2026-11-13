use crate::*;

const SPLINE_OUTPUT_PP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pp",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Piecewise-polynomial struct representation.",
}];

const SPLINE_OUTPUT_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Spline-evaluated values at query points.",
}];

const SPLINE_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
];

const SPLINE_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
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

const SPLINE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pp = spline(X, Y)",
        inputs: &SPLINE_INPUTS_X_Y,
        outputs: &SPLINE_OUTPUT_PP,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = spline(X, Y, Xq)",
        inputs: &SPLINE_INPUTS_X_Y_XQ,
        outputs: &SPLINE_OUTPUT_VALUES,
    },
];

pub const SPLINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:spline:InvalidArgument"),
    when: "Argument count or interpolation option grammar is invalid.",
    message: "spline: invalid argument",
};

pub const SPLINE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INVALID_INPUT",
    identifier: Some("RunMat:spline:InvalidInput"),
    when: "Sample/query inputs are invalid for spline construction or evaluation.",
    message: "spline: invalid input",
};

pub const SPLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLINE.INTERNAL",
    identifier: Some("RunMat:spline:Internal"),
    when: "Spline coefficient generation or tensor assembly fails internally.",
    message: "spline: internal interpolation failure",
};

const SPLINE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SPLINE_ERROR_INVALID_ARGUMENT,
    SPLINE_ERROR_INVALID_INPUT,
    SPLINE_ERROR_INTERNAL,
];

pub const SPLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPLINE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPLINE_ERRORS,
};
