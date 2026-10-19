use crate::*;

pub const LOWER_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "lower-string-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "lower with string scalars inside a cell array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:LowerStringCellExtension"),
};
pub const LOWER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [LOWER_STRING_CELL_EXTENSION];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lowercased text preserving the input container and element shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String, character array, or cell array of text scalars.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = lower(str)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const LOWER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.INVALID_INPUT", identifier: Some("RunMat:lower:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: "lower: first argument must be a string array, character array, or cell array of character vectors",
};
pub const LOWER_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.CELL_ELEMENT",
    identifier: Some("RunMat:lower:CellElement"),
    when: "Cell array contains a non-text element or non-row char array element.",
    message: "lower: cell array elements must be string scalars or character vectors",
};
pub const LOWER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOWER.INTERNAL",
    identifier: Some("RunMat:lower:InternalError"),
    when: "Internal output container construction failed.",
    message: "lower: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    LOWER_ERROR_INVALID_INPUT,
    LOWER_ERROR_CELL_ELEMENT,
    LOWER_ERROR_INTERNAL,
];
pub const LOWER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const LOWER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "lower accepts string arrays, character arrays, or cell arrays of character vectors. Numeric and integer inputs reject without implicit text conversion or provider access.",
};
