use crate::*;
pub const SPLIT_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "newStr",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String or cell array containing split tokens.",
    },
    BuiltinParamDescriptor {
        name: "match",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "String or cell array containing the delimiters at which splitting occurred.",
    },
];

pub const SPLIT_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text scalar/array/cell to split.",
}];

pub const SPLIT_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
];

pub const SPLIT_INPUTS_DELIMITER_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Positive dimension along which output substrings are oriented.",
    },
];

pub const SPLIT_INPUTS_DELIMITER_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `IncludeDelimiters`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

pub const SPLIT_INPUTS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text scalar/array/cell to split.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `IncludeDelimiters`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

pub const SPLIT_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "newStr = split(str)",
        inputs: &SPLIT_INPUTS_BASE,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, delimiter)",
        inputs: &SPLIT_INPUTS_DELIMITER,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[newStr, match] = split(str, delimiter, dim)",
        inputs: &SPLIT_INPUTS_DELIMITER_DIM,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, delimiter, Name, Value, ...)",
        inputs: &SPLIT_INPUTS_DELIMITER_NAMEVALUE,
        outputs: &SPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "newStr = split(str, Name, Value, ...)",
        inputs: &SPLIT_INPUTS_NAMEVALUE,
        outputs: &SPLIT_OUTPUT,
    },
];

pub const SPLIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.INVALID_INPUT",
    identifier: Some("RunMat:split:InvalidInput"),
    when: "First argument is not a string scalar/array, char array, or cell array of text scalars.",
    message:
        "split: first argument must be a string scalar, string array, character array, or cell array of character vectors",
};

pub const SPLIT_ERROR_DELIMITER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.DELIMITER_TYPE",
    identifier: Some("RunMat:split:DelimiterType"),
    when: "Delimiter input is not a supported text scalar/array/cell.",
    message:
        "split: delimiter input must be a string scalar, string array, character array, or cell array of character vectors",
};

pub const SPLIT_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.NAME_VALUE_PAIR",
    identifier: Some("RunMat:split:NameValuePair"),
    when: "Name-value options are not supplied in complete pairs.",
    message: "split: name-value arguments must be supplied in pairs",
};

pub const SPLIT_ERROR_UNKNOWN_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.UNKNOWN_NAME",
    identifier: Some("RunMat:split:UnknownName"),
    when: "An option name is not recognized.",
    message:
        "split: unrecognized name-value argument; supported names are 'CollapseDelimiters' and 'IncludeDelimiters'",
};

pub const SPLIT_ERROR_EMPTY_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.EMPTY_DELIMITER",
    identifier: Some("RunMat:split:EmptyDelimiter"),
    when: "Delimiter list is empty or contains empty delimiter entries.",
    message: "split: delimiters must contain at least one character",
};

pub const SPLIT_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.CELL_ELEMENT",
    identifier: Some("RunMat:split:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "split: cell array elements must be string scalars or character vectors",
};

pub const SPLIT_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.OPTION_VALUE",
    identifier: Some("RunMat:split:OptionValue"),
    when: "Option values are not logical true/false values.",
    message: "split: option values must be logical true or false",
};

pub const SPLIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.INTERNAL",
    identifier: Some("RunMat:split:InternalError"),
    when: "Internal output container construction failed.",
    message: "split: internal error",
};

pub const SPLIT_ERROR_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.DIMENSION",
    identifier: Some("RunMat:split:Dimension"),
    when: "The dimension argument is not a positive integer scalar in RunMat's supported rank range of 1 through 1024.",
    message: "split: dimension must be a positive integer scalar no greater than 1024",
};

pub const SPLIT_ERRORS: [BuiltinErrorDescriptor; 10] = [
    SPLIT_ERROR_TOO_MANY_OUTPUTS,
    SPLIT_ERROR_INVALID_INPUT,
    SPLIT_ERROR_DELIMITER_TYPE,
    SPLIT_ERROR_NAME_VALUE_PAIR,
    SPLIT_ERROR_UNKNOWN_NAME,
    SPLIT_ERROR_EMPTY_DELIMITER,
    SPLIT_ERROR_CELL_ELEMENT,
    SPLIT_ERROR_OPTION_VALUE,
    SPLIT_ERROR_DIMENSION,
    SPLIT_ERROR_INTERNAL,
];

pub const SPLIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SPLIT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SPLIT_ERRORS,
};

pub const SPLIT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPLIT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:split:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "split: too many output arguments",
};

pub const SPLIT_TYPED_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "split-typed-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "split with a typed-integer dimension",
        error_identifier: Some("RunMat:compatibility:SplitTypedDimensionExtension"),
    };

pub const SPLIT_RESIDENT_DIMENSION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "split-resident-dimension",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "split with an explicitly GPU-resident dimension is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SplitResidentDimensionExtension"),
    };

pub const SPLIT_ADVANCED_OPTIONS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "split-advanced-options",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "split CollapseDelimiters and IncludeDelimiters options are RunMat extensions",
        error_identifier: Some("RunMat:compatibility:SplitAdvancedOptionsExtension"),
    };

pub const SPLIT_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SPLIT_TYPED_DIMENSION_EXTENSION,
    SPLIT_RESIDENT_DIMENSION_EXTENSION,
    SPLIT_ADVANCED_OPTIONS_EXTENSION,
];

pub const SPLIT_INTEGER_DIMENSION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target specifies a positive integer dimension but does not enumerate typed storage classes. RunMat accepts every exact integer class behind a compatibility gate; ordinary host double integer dimensions remain documented behavior.",
    }];

pub const SPLIT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[newStr, match] = split(str, delimiter, integer_dim)",
        inputs: &SPLIT_INTEGER_DIMENSION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The typed dimension is a gated RunMat extension whose exact one-based value controls only output orientation. Explicit resident dimensions are separately gated before gather; automatic residency may gather transparently.",
    }];
