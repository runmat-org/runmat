use crate::*;

pub const STRTRIM_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "strtrim-string-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "strtrim with string scalars inside a cell array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:StrtrimStringCellExtension"),
};
pub const STRTRIM_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [STRTRIM_STRING_CELL_EXTENSION];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Trimmed text preserving the input container and element shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String, character array, or cell array of text scalars to trim.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = strtrim(str)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRTRIM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.INVALID_INPUT", identifier: Some("RunMat:strtrim:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: "strtrim: first argument must be a string array, character array, or cell array of character vectors",
};
pub const STRTRIM_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.CELL_ELEMENT",
    identifier: Some("RunMat:strtrim:CellElement"),
    when: "Cell array contains a non-text element or non-row char array element.",
    message: "strtrim: cell array elements must be string scalars or character vectors",
};
pub const STRTRIM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRTRIM.INTERNAL",
    identifier: Some("RunMat:strtrim:InternalError"),
    when: "Internal output container construction failed.",
    message: "strtrim: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    STRTRIM_ERROR_INVALID_INPUT,
    STRTRIM_ERROR_CELL_ELEMENT,
    STRTRIM_ERROR_INTERNAL,
];
pub const STRTRIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const STRTRIM_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "strtrim removes whitespace from text containers. Numeric, integer, and provider-resident values reject before provider access and are not interpreted as character codes.",
};
