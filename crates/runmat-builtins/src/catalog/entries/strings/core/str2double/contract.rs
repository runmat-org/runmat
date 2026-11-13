use crate::*;

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String, character, or cell-array text to parse.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double values; invalid numeric text becomes NaN.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "X = str2double(str)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STR2DOUBLE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2DOUBLE.INVALID_INPUT",
    identifier: Some("RunMat:str2double:InvalidInput"),
    when: "Input is not a supported text container.",
    message: "str2double: input must be a string array, character array, or cell array of character vectors",
};
pub const STR2DOUBLE_ERROR_INVALID_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2DOUBLE.INVALID_CELL_ELEMENT",
    identifier: Some("RunMat:str2double:InvalidCellElement"),
    when: "Cell array contains non-text or non-scalar text entries.",
    message: "str2double: cell array elements must be character vectors or string scalars",
};
pub const STR2DOUBLE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2DOUBLE.INTERNAL",
    identifier: Some("RunMat:str2double:InternalError"),
    when: "Internal tensor assembly failed while building parsed output.",
    message: "str2double: internal error",
};
pub const STR2DOUBLE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STR2DOUBLE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:str2double:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "str2double: too many output arguments",
};
pub const STR2DOUBLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STR2DOUBLE_ERROR_INVALID_INPUT,
        STR2DOUBLE_ERROR_INVALID_CELL_ELEMENT,
        STR2DOUBLE_ERROR_INTERNAL,
        STR2DOUBLE_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const STR2DOUBLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Text inputs parse to double. Numeric and integer inputs, including resident numeric values, reject before provider access.",
};
