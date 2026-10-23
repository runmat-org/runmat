use crate::*;
pub const STRCAT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Concatenated text preserving strcat output container semantics.",
}];

pub const STRCAT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First text input (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "str2",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional text inputs to concatenate element-wise.",
    },
];

pub const STRCAT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strcat(str1, str2, ...)",
    inputs: &STRCAT_INPUTS,
    outputs: &STRCAT_OUTPUT,
}];

pub const STRCAT_ERROR_NOT_ENOUGH_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.NOT_ENOUGH_INPUTS",
    identifier: Some("RunMat:strcat:NotEnoughInputs"),
    when: "No arguments are supplied.",
    message: "strcat: not enough input arguments",
};

pub const STRCAT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.INVALID_INPUT",
    identifier: Some("RunMat:strcat:InvalidInput"),
    when: "An input is not a string, character array, or cell array of text scalars.",
    message:
        "strcat: inputs must be strings, character arrays, or cell arrays of character vectors",
};

pub const STRCAT_ERROR_INVALID_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.CELL_ELEMENT",
    identifier: Some("RunMat:strcat:CellElement"),
    when: "A cell array contains a non-text element or non-row char array element.",
    message: "strcat: cell array elements must be character vectors or string scalars",
};

pub const STRCAT_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.SIZE_MISMATCH",
    identifier: Some("RunMat:strcat:SizeMismatch"),
    when: "Input shapes are not broadcast-compatible.",
    message: "strcat: array sizes are not compatible for broadcasting",
};

pub const STRCAT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.INTERNAL",
    identifier: Some("RunMat:strcat:InternalError"),
    when: "Internal output container construction failed.",
    message: "strcat: internal error",
};

pub const STRCAT_ERRORS: [BuiltinErrorDescriptor; 6] = [
    STRCAT_ERROR_TOO_MANY_OUTPUTS,
    STRCAT_ERROR_NOT_ENOUGH_INPUTS,
    STRCAT_ERROR_INVALID_INPUT,
    STRCAT_ERROR_INVALID_CELL_ELEMENT,
    STRCAT_ERROR_SIZE_MISMATCH,
    STRCAT_ERROR_INTERNAL,
];

pub const STRCAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRCAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRCAT_ERRORS,
};

pub const STRCAT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strcat concatenates character arrays, string arrays, and cell arrays of character vectors. Numeric, integer, and provider-resident values reject before provider access without implicit text conversion.",
};

pub const STRCAT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCAT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strcat:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strcat: too many output arguments",
};
