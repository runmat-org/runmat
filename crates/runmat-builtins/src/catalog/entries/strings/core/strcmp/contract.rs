use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical comparison result.",
}];
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First string, character array, or cellstr operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second string, character array, or cellstr operand.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = strcmp(A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRCMP_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMP.INVALID_INPUT",
    identifier: Some("RunMat:strcmp:InvalidInput"),
    when: "At least one input is not a supported text container.",
    message: "strcmp: text inputs must be string/char/cell/string-array values",
};
pub const STRCMP_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMP.SHAPE_MISMATCH",
    identifier: Some("RunMat:strcmp:ShapeMismatch"),
    when: "Inputs are not broadcast-compatible for elementwise comparison.",
    message: "strcmp: input sizes are not broadcast-compatible",
};
pub const STRCMP_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMP.INTERNAL",
    identifier: Some("RunMat:strcmp:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "strcmp: internal error",
};
pub const STRCMP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strcmp:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strcmp: too many output arguments",
};
pub const STRCMP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STRCMP_ERROR_INVALID_INPUT,
        STRCMP_ERROR_SHAPE_MISMATCH,
        STRCMP_ERROR_INTERNAL,
        STRCMP_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const STRCMP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "strcmp compares text containers. An unsupported direct integer or numeric input returns scalar logical false without provider access; malformed host cell elements are diagnosed by the container validator.",
};
