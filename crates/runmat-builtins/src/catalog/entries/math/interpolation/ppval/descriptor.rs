use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Evaluated piecewise-polynomial values at query points.",
}];

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "pp",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Piecewise-polynomial struct with form/breaks/coefs fields.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Vq = ppval(pp, Xq)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INVALID_ARGUMENT",
    identifier: Some("RunMat:ppval:InvalidArgument"),
    when: "The piecewise-polynomial struct shape or field grammar is invalid.",
    message: "ppval: invalid argument",
};

pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INVALID_INPUT",
    identifier: Some("RunMat:ppval:InvalidInput"),
    when: "Query points or polynomial coefficients are invalid for evaluation.",
    message: "ppval: invalid input",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PPVAL.INTERNAL",
    identifier: Some("RunMat:ppval:Internal"),
    when: "Piecewise-polynomial output assembly fails due to internal tensor construction paths.",
    message: "ppval: internal interpolation failure",
};

const ERRORS: [BuiltinErrorDescriptor; 3] =
    [ERROR_INVALID_ARGUMENT, ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const PPVAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
