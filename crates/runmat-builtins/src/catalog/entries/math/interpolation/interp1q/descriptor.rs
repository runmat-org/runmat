use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Linearly interpolated query values.",
}];

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Strictly increasing sample locations.",
    },
    BuiltinParamDescriptor {
        name: "V",
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
        description: "Query locations.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Vq = interp1q(X, V, Xq)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INVALID_INPUT",
    identifier: Some("RunMat:interp1q:InvalidInput"),
    when:
        "Arguments cannot be converted to the legacy one-dimensional linear interpolation domain.",
    message: "interp1q: invalid input",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1Q.INTERNAL",
    identifier: Some("RunMat:interp1q:Internal"),
    when: "Interpolation evaluation or output construction fails unexpectedly.",
    message: "interp1q: internal interpolation failure",
};

const ERRORS: [BuiltinErrorDescriptor; 2] = [ERROR_INVALID_INPUT, ERROR_INTERNAL];

pub const INTERP1Q_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
