use crate::*;

const PCHIP_OUTPUT_PP: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pp",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Piecewise-polynomial struct representation.",
}];

const PCHIP_OUTPUT_VALUES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "PCHIP-evaluated values at query points.",
}];

const PCHIP_INPUTS_X_Y: [BuiltinParamDescriptor; 2] = [
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

const PCHIP_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
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

const PCHIP_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "pp = pchip(X, Y)",
        inputs: &PCHIP_INPUTS_X_Y,
        outputs: &PCHIP_OUTPUT_PP,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = pchip(X, Y, Xq)",
        inputs: &PCHIP_INPUTS_X_Y_XQ,
        outputs: &PCHIP_OUTPUT_VALUES,
    },
];

pub const PCHIP_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INVALID_ARGUMENT",
    identifier: Some("RunMat:pchip:InvalidArgument"),
    when: "Argument count or interpolation option grammar is invalid.",
    message: "pchip: invalid argument",
};

pub const PCHIP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INVALID_INPUT",
    identifier: Some("RunMat:pchip:InvalidInput"),
    when: "Sample/query inputs are invalid for PCHIP construction or evaluation.",
    message: "pchip: invalid input",
};

pub const PCHIP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PCHIP.INTERNAL",
    identifier: Some("RunMat:pchip:Internal"),
    when: "PCHIP coefficient generation or tensor assembly fails internally.",
    message: "pchip: internal interpolation failure",
};

const PCHIP_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PCHIP_ERROR_INVALID_ARGUMENT,
    PCHIP_ERROR_INVALID_INPUT,
    PCHIP_ERROR_INTERNAL,
];

pub const PCHIP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PCHIP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PCHIP_ERRORS,
};
