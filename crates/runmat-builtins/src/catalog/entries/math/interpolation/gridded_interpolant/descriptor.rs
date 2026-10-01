use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "F",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Gridded interpolant object.",
}];

const SUBSREF_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Indexed value or property.",
}];

const INPUTS_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description:
        "Grid vectors or full grids, values, interpolation method, and extrapolation method.",
}];

const SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "griddedInterpolant object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant",
        inputs: &[],
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(x, v)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(gridVecs, V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(X1, X2, ..., Xn, V)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(___, Method)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "F = griddedInterpolant(___, Method, ExtrapolationMethod)",
        inputs: &INPUTS_VARIADIC,
        outputs: &OUTPUT,
    },
];

const SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = griddedInterpolant.subsref(obj, S)",
    inputs: &SUBSREF_INPUTS,
    outputs: &SUBSREF_OUTPUT,
}];

pub const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.INVALID_INPUT",
    identifier: Some("RunMat:griddedInterpolant:InvalidInput"),
    when: "Grid, value, method, or query inputs are invalid.",
    message: "griddedInterpolant: invalid input",
};

pub const ERROR_UNSUPPORTED_METHOD: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.UNSUPPORTED_METHOD",
    identifier: Some("RunMat:griddedInterpolant:UnsupportedMethod"),
    when: "The requested interpolation or extrapolation method is not implemented by RunMat.",
    message: "griddedInterpolant: unsupported interpolation method",
};

pub const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GRIDDEDINTERPOLANT.INTERNAL",
    identifier: Some("RunMat:griddedInterpolant:Internal"),
    when: "Object construction, property access, or interpolation output construction fails.",
    message: "griddedInterpolant: internal error",
};

const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ERROR_INVALID_INPUT,
    ERROR_UNSUPPORTED_METHOD,
    ERROR_INTERNAL,
];

pub const GRIDDED_INTERPOLANT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const GRIDDED_INTERPOLANT_SUBSREF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUBSREF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: &ERRORS,
};
