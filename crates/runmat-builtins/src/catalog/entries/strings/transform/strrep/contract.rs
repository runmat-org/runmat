use crate::*;
pub const STRREP_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "newStr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text with pattern occurrences replaced, preserving input container kind.",
}];

pub const STRREP_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input text (string/char/cell).",
    },
    BuiltinParamDescriptor {
        name: "old",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text scalar (string or char row).",
    },
    BuiltinParamDescriptor {
        name: "new",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Replacement text scalar matching old's data type family.",
    },
];

pub const STRREP_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "newStr = strrep(str, old, new)",
    inputs: &STRREP_INPUTS,
    outputs: &STRREP_OUTPUT,
}];

pub const STRREP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.INVALID_INPUT",
    identifier: Some("RunMat:strrep:InvalidInput"),
    when: "First argument is not a string array, char array, or cell array of text scalars.",
    message:
        "strrep: first argument must be a string array, character array, or cell array of character vectors",
};

pub const STRREP_ERROR_PATTERN_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.PATTERN_TYPE",
    identifier: Some("RunMat:strrep:PatternType"),
    when: "old/new arguments are not string scalars or character vectors.",
    message: "strrep: old and new must be string scalars or character vectors",
};

pub const STRREP_ERROR_PATTERN_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.PATTERN_MISMATCH",
    identifier: Some("RunMat:strrep:PatternMismatch"),
    when: "old and new are different text data families (string vs char).",
    message: "strrep: old and new must be the same data type",
};

pub const STRREP_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.CELL_ELEMENT",
    identifier: Some("RunMat:strrep:CellElement"),
    when: "Cell input contains non-text elements or non-row char arrays.",
    message: "strrep: cell array elements must be string scalars or character vectors",
};

pub const STRREP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.INTERNAL",
    identifier: Some("RunMat:strrep:InternalError"),
    when: "Internal output container construction failed.",
    message: "strrep: internal error",
};

pub const STRREP_ERRORS: [BuiltinErrorDescriptor; 6] = [
    STRREP_ERROR_TOO_MANY_OUTPUTS,
    STRREP_ERROR_INVALID_INPUT,
    STRREP_ERROR_PATTERN_TYPE,
    STRREP_ERROR_PATTERN_MISMATCH,
    STRREP_ERROR_CELL_ELEMENT,
    STRREP_ERROR_INTERNAL,
];

pub const STRREP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRREP_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRREP_ERRORS,
};

pub const STRREP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "strrep accepts text in the subject, old-pattern, and replacement roles. Numeric, integer, and provider-resident values reject before provider access and are never interpreted as character codes.",
};

pub const STRREP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRREP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strrep:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strrep: too many output arguments",
};
