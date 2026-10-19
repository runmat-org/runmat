use crate::*;

pub const UPPER_STRING_CELL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "upper-string-cell",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "upper with string scalars inside a cell array is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:UpperStringCellExtension"),
};
pub const UPPER_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [UPPER_STRING_CELL_EXTENSION];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Uppercased text preserving the input container and element shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String, character array, or cell array of text scalars.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "out = upper(str)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const UPPER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UPPER.INVALID_INPUT", identifier: Some("RunMat:upper:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: "upper: first argument must be a string array, character array, or cell array of character vectors",
};
pub const UPPER_ERROR_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UPPER.CELL_ELEMENT",
    identifier: Some("RunMat:upper:CellElement"),
    when: "Cell array contains a non-text element or non-row char array element.",
    message: "upper: cell array elements must be string scalars or character vectors",
};
pub const UPPER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UPPER.INTERNAL",
    identifier: Some("RunMat:upper:InternalError"),
    when: "Internal output container construction failed.",
    message: "upper: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    UPPER_ERROR_INVALID_INPUT,
    UPPER_ERROR_CELL_ELEMENT,
    UPPER_ERROR_INTERNAL,
];
pub const UPPER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const UPPER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "upper accepts string arrays, character arrays, or cell arrays of character vectors. Numeric and integer inputs reject without implicit text conversion or provider access.",
};
