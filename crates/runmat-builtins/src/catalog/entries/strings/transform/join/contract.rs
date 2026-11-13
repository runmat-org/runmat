use crate::*;
pub const JOIN_TYPED_INTEGER_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "join-typed-integer-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "join with a typed-integer dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JoinTypedIntegerDimensionExtension"),
    };

pub const JOIN_REVERSED_ARGUMENTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "join-dimension-before-delimiter",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "join(str, dim, delimiter) is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:JoinDimensionBeforeDelimiterExtension"),
    };

pub const JOIN_RESIDENT_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "join-resident-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "join with an explicitly resident gpuArray argument is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:JoinResidentInputExtension"),
};

pub const JOIN_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    JOIN_TYPED_INTEGER_DIMENSION_EXTENSION,
    JOIN_REVERSED_ARGUMENTS_EXTENSION,
    JOIN_RESIDENT_INPUT_EXTENSION,
];

pub const JOIN_INTEGER_DIMENSION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented dimension is a positive integer-valued double scalar. RunMat mode additionally accepts every native typed-integer scalar exactly.",
    }];

pub const JOIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "out = join(str, delimiter?, integer_dim)",
        inputs: &JOIN_INTEGER_DIMENSION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Typed dimensions are compatibility-gated before resident access, parsed from authoritative integer storage, and never converted through floating point.",
    }];

pub const JOIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Joined text preserving join output container semantics.",
}];

pub const JOIN_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text (string/char/cell).",
}];

pub const JOIN_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("\" \""),
        description: "Delimiter scalar or delimiter array matching join shape constraints.",
    },
];

pub const JOIN_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive dimension index to join along.",
    },
];

pub const JOIN_INPUTS_DELIMITER_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar or delimiter array matching join shape constraints.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive dimension index to join along.",
    },
];

pub const JOIN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "out = join(str)",
        inputs: &JOIN_INPUTS_BASE,
        outputs: &JOIN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = join(str, delimiter)",
        inputs: &JOIN_INPUTS_DELIMITER,
        outputs: &JOIN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = join(str, dim)",
        inputs: &JOIN_INPUTS_DIM,
        outputs: &JOIN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "out = join(str, delimiter, dim)",
        inputs: &JOIN_INPUTS_DELIMITER_DIM,
        outputs: &JOIN_OUTPUT,
    },
];

pub const JOIN_ERROR_INPUT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.INPUT_TYPE",
    identifier: Some("RunMat:join:InputType"),
    when: "Input text is not a string array/scalar, char array, or cell array of text scalars.",
    message:
        "join: input must be a string array, string scalar, character array, or cell array of character vectors",
};

pub const JOIN_ERROR_DELIMITER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.DELIMITER_TYPE",
    identifier: Some("RunMat:join:DelimiterType"),
    when: "Delimiter is not a supported text scalar/array/cell value.",
    message:
        "join: delimiter must be a string, character vector, string array, or cell array of character vectors",
};

pub const JOIN_ERROR_DELIMITER_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.DELIMITER_SIZE",
    identifier: Some("RunMat:join:DelimiterSize"),
    when: "Delimiter array shape does not match join shape constraints.",
    message:
        "join: size of delimiter array must match the size of str, with the join dimension reduced by one",
};

pub const JOIN_ERROR_DIMENSION_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.DIMENSION_TYPE",
    identifier: Some("RunMat:join:DimensionType"),
    when: "Dimension argument is not a positive integer scalar.",
    message: "join: dimension must be a positive integer scalar",
};

pub const JOIN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.ARG_COUNT",
    identifier: Some("RunMat:join:ArgCount"),
    when: "More than three total arguments are supplied.",
    message: "join: too many input arguments",
};

pub const JOIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.INTERNAL",
    identifier: Some("RunMat:join:InternalError"),
    when: "Internal output container construction failed.",
    message: "join: internal error",
};

pub const JOIN_ERRORS: [BuiltinErrorDescriptor; 7] = [
    JOIN_ERROR_TOO_MANY_OUTPUTS,
    JOIN_ERROR_INPUT_TYPE,
    JOIN_ERROR_DELIMITER_TYPE,
    JOIN_ERROR_DELIMITER_SIZE,
    JOIN_ERROR_DIMENSION_TYPE,
    JOIN_ERROR_ARG_COUNT,
    JOIN_ERROR_INTERNAL,
];

pub const JOIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &JOIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &JOIN_ERRORS,
};

pub const JOIN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.JOIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:join:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "join: too many output arguments",
};
