use crate::*;
pub const PAD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Padded text preserving input container kind and shape.",
}];

pub const PAD_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text (string/char/cell).",
}];

pub const PAD_INPUTS_LENGTH: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
];

pub const PAD_INPUTS_DIRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
];

pub const PAD_INPUTS_PADCHAR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

pub const PAD_INPUTS_LENGTH_DIRECTION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
];

pub const PAD_INPUTS_LENGTH_PADCHAR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

pub const PAD_INPUTS_DIRECTION_PADCHAR: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

pub const PAD_INPUTS_LENGTH_DIRECTION_PADCHAR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "len",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Target length (non-negative integer).",
    },
    BuiltinParamDescriptor {
        name: "direction",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"right\""),
        description: "Padding direction (`\"left\"|\"right\"|\"both\"`).",
    },
    BuiltinParamDescriptor {
        name: "padCharacter",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Single-character padding value.",
    },
];

pub const PAD_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "out = pad(str)",
        inputs: &PAD_INPUTS_BASE,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len)",
        inputs: &PAD_INPUTS_LENGTH,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, direction)",
        inputs: &PAD_INPUTS_DIRECTION,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, padCharacter)",
        inputs: &PAD_INPUTS_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, direction)",
        inputs: &PAD_INPUTS_LENGTH_DIRECTION,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, padCharacter)",
        inputs: &PAD_INPUTS_LENGTH_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, direction, padCharacter)",
        inputs: &PAD_INPUTS_DIRECTION_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = pad(str, len, direction, padCharacter)",
        inputs: &PAD_INPUTS_LENGTH_DIRECTION_PADCHAR,
        outputs: &PAD_OUTPUT,
    },
];

pub const PAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.INVALID_INPUT",
    identifier: Some("RunMat:pad:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message:
        "pad: first argument must be a string array, character array, or cell array of character vectors",
};

pub const PAD_ERROR_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.LENGTH",
    identifier: Some("RunMat:pad:Length"),
    when: "Length argument is not a non-negative integer scalar.",
    message: "pad: target length must be a non-negative integer scalar",
};

pub const PAD_ERROR_DIRECTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.DIRECTION",
    identifier: Some("RunMat:pad:Direction"),
    when: "Direction argument is not one of left/right/both.",
    message: "pad: direction must be 'left', 'right', or 'both'",
};

pub const PAD_ERROR_PAD_CHAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.PAD_CHAR",
    identifier: Some("RunMat:pad:PadChar"),
    when: "Padding character is not a single-character string/char scalar.",
    message:
        "pad: padding character must be a string scalar or character vector containing one character",
};

pub const PAD_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.CELL_ELEMENT",
    identifier: Some("RunMat:pad:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "pad: cell array elements must be string scalars or character vectors",
};

pub const PAD_ERROR_ARGUMENT_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.ARGUMENT_CONFIG",
    identifier: Some("RunMat:pad:ArgumentConfig"),
    when: "Second/third arguments cannot be interpreted as valid pad argument combinations.",
    message: "pad: unable to interpret input arguments",
};

pub const PAD_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.ARG_COUNT",
    identifier: Some("RunMat:pad:ArgCount"),
    when: "More than four total arguments are supplied.",
    message: "pad: too many input arguments",
};

pub const PAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.INTERNAL",
    identifier: Some("RunMat:pad:InternalError"),
    when: "Internal output container construction failed.",
    message: "pad: internal error",
};

pub const PAD_ERRORS: [BuiltinErrorDescriptor; 9] = [
    PAD_ERROR_TOO_MANY_OUTPUTS,
    PAD_ERROR_INVALID_INPUT,
    PAD_ERROR_LENGTH,
    PAD_ERROR_DIRECTION,
    PAD_ERROR_PAD_CHAR,
    PAD_ERROR_CELL_ELEMENT,
    PAD_ERROR_ARGUMENT_CONFIG,
    PAD_ERROR_ARG_COUNT,
    PAD_ERROR_INTERNAL,
];

pub const PAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAD_ERRORS,
};

pub const PAD_LENGTH_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numberOfCharacters",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target explicitly documents single, double, and every built-in integer class for the nonnegative scalar output length.",
    }];

pub const PAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "newStr = pad(str, integer_numberOfCharacters, side?, padCharacter?)",
        inputs: &PAD_LENGTH_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact integer controls only target text length; output preserves the input text container. Explicit interactive GPU length input is unsupported, while automatic residency may gather transparently.",
    }];

pub const PAD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pad:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "pad: too many output arguments",
};
