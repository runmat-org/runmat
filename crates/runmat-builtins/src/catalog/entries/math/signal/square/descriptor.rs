use crate::*;
const SQUARE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square-wave output sampled at t.",
}];

const SQUARE_SIG_DEFAULT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sample times.",
}];

const SQUARE_SIG_DUTY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample times.",
    },
    BuiltinParamDescriptor {
        name: "duty",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("50"),
        description: "Duty cycle percentage in [0, 100].",
    },
];

const SQUARE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = square(t)",
        inputs: &SQUARE_SIG_DEFAULT_INPUTS,
        outputs: &SQUARE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = square(t, duty)",
        inputs: &SQUARE_SIG_DUTY_INPUTS,
        outputs: &SQUARE_OUTPUT,
    },
];

pub const SQUARE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.INVALID_INPUT",
    identifier: Some("RunMat:square:InvalidInput"),
    when: "Primary input is not numeric-real tensor/scalar compatible.",
    message: "square: expected numeric input",
};

pub const SQUARE_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:square:ComplexUnsupported"),
    when: "Primary input includes complex values.",
    message: "square: input must be real; complex values are not supported",
};

pub const SQUARE_ERROR_DUTY_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.DUTY_INVALID",
    identifier: Some("RunMat:square:DutyInvalid"),
    when: "Duty argument is not a real numeric scalar.",
    message: "square: duty must be a real numeric scalar in [0, 100]",
};

pub const SQUARE_ERROR_DUTY_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.DUTY_RANGE",
    identifier: Some("RunMat:square:DutyOutOfRange"),
    when: "Duty argument lies outside [0, 100] or is non-finite.",
    message: "square: duty must be a finite scalar in [0, 100]",
};

pub const SQUARE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.ARG_COUNT",
    identifier: Some("RunMat:square:ArgCount"),
    when: "More than one optional argument is provided.",
    message: "square: expected 1 or 2 arguments",
};

pub const SQUARE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SQUARE.INTERNAL",
    identifier: Some("RunMat:square:InternalError"),
    when: "Internal tensor construction or GPU gather fails.",
    message: "square: internal error",
};

const SQUARE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SQUARE_ERROR_INVALID_INPUT,
    SQUARE_ERROR_COMPLEX_UNSUPPORTED,
    SQUARE_ERROR_DUTY_INVALID,
    SQUARE_ERROR_DUTY_RANGE,
    SQUARE_ERROR_ARG_COUNT,
    SQUARE_ERROR_INTERNAL,
];

pub const SQUARE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SQUARE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SQUARE_ERRORS,
};
