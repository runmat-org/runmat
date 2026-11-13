use crate::*;
pub const STRSPLIT_OUTPUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "parts",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Split tokens.",
    },
    BuiltinParamDescriptor {
        name: "matches",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Matched delimiters when requested as second output.",
    },
];

pub const STRSPLIT_INPUTS_BASE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar or character vector input.",
}];

pub const STRSPLIT_INPUTS_DELIMITER: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
    },
    BuiltinParamDescriptor {
        name: "delimiter",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Delimiter scalar/array/cell.",
    },
];

pub const STRSPLIT_INPUTS_DELIMITER_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
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
        description: "Option name (`CollapseDelimiters` or `DelimiterType`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

pub const STRSPLIT_INPUTS_NAMEVALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "String scalar or character vector input.",
    },
    BuiltinParamDescriptor {
        name: "Name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option name (`CollapseDelimiters` or `DelimiterType`).",
    },
    BuiltinParamDescriptor {
        name: "Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option values and additional Name/Value pairs.",
    },
];

pub const STRSPLIT_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str)",
        inputs: &STRSPLIT_INPUTS_BASE,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, delimiter)",
        inputs: &STRSPLIT_INPUTS_DELIMITER,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, delimiter, Name, Value, ...)",
        inputs: &STRSPLIT_INPUTS_DELIMITER_NAMEVALUE,
        outputs: &STRSPLIT_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[parts, matches] = strsplit(str, Name, Value, ...)",
        inputs: &STRSPLIT_INPUTS_NAMEVALUE,
        outputs: &STRSPLIT_OUTPUT,
    },
];

pub const STRSPLIT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.INVALID_INPUT",
    identifier: Some("RunMat:strsplit:InvalidInput"),
    when: "First argument is not a string scalar or character vector.",
    message: "strsplit: first argument must be a string scalar or character vector",
};

pub const STRSPLIT_ERROR_DELIMITER_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.DELIMITER_TYPE",
    identifier: Some("RunMat:strsplit:DelimiterType"),
    when: "Delimiter input is not a supported text scalar/array/cell.",
    message:
        "strsplit: delimiter must be a character vector, string scalar, string array, or cell array of character vectors",
};

pub const STRSPLIT_ERROR_NAME_VALUE_PAIR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.NAME_VALUE_PAIR",
    identifier: Some("RunMat:strsplit:NameValuePair"),
    when: "Name-value options are not supplied in complete pairs.",
    message: "strsplit: name-value arguments must be supplied in pairs",
};

pub const STRSPLIT_ERROR_UNKNOWN_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.UNKNOWN_NAME",
    identifier: Some("RunMat:strsplit:UnknownName"),
    when: "An option name is not recognized.",
    message:
        "strsplit: unrecognized name-value argument; supported names are 'CollapseDelimiters' and 'DelimiterType'",
};

pub const STRSPLIT_ERROR_EMPTY_DELIMITER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.EMPTY_DELIMITER",
    identifier: Some("RunMat:strsplit:EmptyDelimiter"),
    when: "Delimiter list is empty or contains empty delimiter entries.",
    message: "strsplit: delimiters must contain at least one character",
};

pub const STRSPLIT_ERROR_DELIMITER_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.DELIMITER_MODE",
    identifier: Some("RunMat:strsplit:DelimiterMode"),
    when: "DelimiterType option is not `Simple` or `RegularExpression`.",
    message: "strsplit: value for 'DelimiterType' must be 'Simple' or 'RegularExpression'",
};

pub const STRSPLIT_ERROR_OPTION_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.OPTION_VALUE",
    identifier: Some("RunMat:strsplit:OptionValue"),
    when: "Option values are not logical true/false values.",
    message: "strsplit: option values must be logical true or false",
};

pub const STRSPLIT_ERROR_REGEX_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.REGEX_INVALID",
    identifier: Some("RunMat:strsplit:RegexInvalid"),
    when: "Regular expression delimiter pattern fails to compile.",
    message: "strsplit: invalid delimiter regular expression",
};

pub const STRSPLIT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.INTERNAL",
    identifier: Some("RunMat:strsplit:InternalError"),
    when: "Internal output container construction failed.",
    message: "strsplit: internal error",
};

pub const STRSPLIT_ERRORS: [BuiltinErrorDescriptor; 10] = [
    STRSPLIT_ERROR_TOO_MANY_OUTPUTS,
    STRSPLIT_ERROR_INVALID_INPUT,
    STRSPLIT_ERROR_DELIMITER_TYPE,
    STRSPLIT_ERROR_NAME_VALUE_PAIR,
    STRSPLIT_ERROR_UNKNOWN_NAME,
    STRSPLIT_ERROR_EMPTY_DELIMITER,
    STRSPLIT_ERROR_DELIMITER_MODE,
    STRSPLIT_ERROR_OPTION_VALUE,
    STRSPLIT_ERROR_REGEX_INVALID,
    STRSPLIT_ERROR_INTERNAL,
];

pub const STRSPLIT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRSPLIT_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRSPLIT_ERRORS,
};

pub const STRSPLIT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRSPLIT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strsplit:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "strsplit: too many output arguments",
};

pub const STRSPLIT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strsplit accepts scalar text, text delimiters, and textual or logical options. Integer and provider-resident numeric values have no documented role and reject before provider access without implicit character conversion.",
};
