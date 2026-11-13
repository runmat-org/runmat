use crate::*;
pub const REPLACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newText",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with replacements applied, preserving input container kind.",
}];

pub const REPLACE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "oldText",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Search text list (scalar or array/cell).",
    },
    BuiltinParamDescriptor {
        name: "newText",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement text list (scalar or matching-size list).",
    },
];

pub const REPLACE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "newText = replace(str, oldText, newText)",
    inputs: &REPLACE_INPUTS,
    outputs: &REPLACE_OUTPUT,
}];

pub const REPLACE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.INVALID_INPUT",
    identifier: Some("RunMat:replace:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message:
        "replace: first argument must be a string array, character array, or cell array of character vectors",
};

pub const REPLACE_ERROR_PATTERN_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.PATTERN_TYPE",
    identifier: Some("RunMat:replace:PatternType"),
    when: "Second argument is not a text scalar/array/cell of text scalars.",
    message:
        "replace: second argument must be a string array, character array, or cell array of character vectors",
};

pub const REPLACE_ERROR_REPLACEMENT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.REPLACEMENT_TYPE",
    identifier: Some("RunMat:replace:ReplacementType"),
    when: "Third argument is not a text scalar/array/cell of text scalars.",
    message:
        "replace: third argument must be a string array, character array, or cell array of character vectors",
};

pub const REPLACE_ERROR_EMPTY_PATTERN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.EMPTY_PATTERN",
    identifier: Some("RunMat:replace:EmptyPattern"),
    when: "Search text list is empty.",
    message: "replace: second argument must contain at least one search string",
};

pub const REPLACE_ERROR_EMPTY_REPLACEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.EMPTY_REPLACEMENT",
    identifier: Some("RunMat:replace:EmptyReplacement"),
    when: "Replacement text list is empty.",
    message: "replace: third argument must contain at least one replacement string",
};

pub const REPLACE_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.SIZE_MISMATCH",
    identifier: Some("RunMat:replace:SizeMismatch"),
    when: "Replacement list is neither scalar nor equal in length to search list.",
    message: "replace: replacement array must be a scalar or match the number of search strings",
};

pub const REPLACE_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.CELL_ELEMENT",
    identifier: Some("RunMat:replace:CellElement"),
    when: "Cell arrays contain non-text elements or non-row char arrays.",
    message: "replace: cell array elements must be string scalars or character vectors",
};

pub const REPLACE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.INTERNAL",
    identifier: Some("RunMat:replace:InternalError"),
    when: "Internal output container construction failed.",
    message: "replace: internal error",
};

pub const REPLACE_ERRORS: [BuiltinErrorDescriptor; 9] = [
    REPLACE_ERROR_TOO_MANY_OUTPUTS,
    REPLACE_ERROR_INVALID_INPUT,
    REPLACE_ERROR_PATTERN_TYPE,
    REPLACE_ERROR_REPLACEMENT_TYPE,
    REPLACE_ERROR_EMPTY_PATTERN,
    REPLACE_ERROR_EMPTY_REPLACEMENT,
    REPLACE_ERROR_SIZE_MISMATCH,
    REPLACE_ERROR_CELL_ELEMENT,
    REPLACE_ERROR_INTERNAL,
];

pub const REPLACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &REPLACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &REPLACE_ERRORS,
};

pub const REPLACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "replace accepts host string, character, cell text, and pattern inputs only and preserves the input text container. Numeric, logical, symbolic, and provider-resident values reject before gather or provider access.",
};

pub const REPLACE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REPLACE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:replace:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "replace: too many output arguments",
};
