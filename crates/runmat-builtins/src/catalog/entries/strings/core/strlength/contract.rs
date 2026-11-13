use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double character count for each text element.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "str",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar or array, character array, or cell array of text scalars.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "L = strlength(str)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRLENGTH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRLENGTH.INVALID_INPUT", identifier: Some("RunMat:strlength:InvalidInput"),
    when: "Input is not a string array, character array, or cell array of text scalars.",
    message: "strlength: first argument must be a string array, character array, or cell array of character vectors",
};
pub const STRLENGTH_ERROR_INVALID_CELL_ELEMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRLENGTH.INVALID_CELL_ELEMENT",
    identifier: Some("RunMat:strlength:InvalidCellElement"),
    when: "A cell-array element is not a character row vector or scalar string.",
    message: "strlength: cell array elements must be character vectors or string scalars",
};
pub const STRLENGTH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRLENGTH.INTERNAL",
    identifier: Some("RunMat:strlength:InternalError"),
    when: "Internal tensor construction failed while building length results.",
    message: "strlength: internal error",
};
pub const STRLENGTH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRLENGTH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strlength:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strlength: too many output arguments",
};
pub const STRLENGTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STRLENGTH_ERROR_INVALID_INPUT,
        STRLENGTH_ERROR_INVALID_CELL_ELEMENT,
        STRLENGTH_ERROR_INTERNAL,
        STRLENGTH_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const STRLENGTH_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "strlength measures host text and returns double counts. Integer and resident numeric inputs reject before provider access; they are never interpreted as character codes.",
};
