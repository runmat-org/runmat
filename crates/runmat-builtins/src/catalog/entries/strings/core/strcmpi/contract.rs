use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Case-insensitive logical comparison result.",
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
    label: "tf = strcmpi(A, B)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const STRCMPI_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.INVALID_INPUT",
    identifier: Some("RunMat:strcmpi:InvalidInput"),
    when: "At least one input is not a supported text container.",
    message: "strcmpi: text inputs must be string/char/cell/string-array values",
};
pub const STRCMPI_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.SHAPE_MISMATCH",
    identifier: Some("RunMat:strcmpi:ShapeMismatch"),
    when: "Inputs are not broadcast-compatible for elementwise comparison.",
    message: "strcmpi: input sizes are not broadcast-compatible",
};
pub const STRCMPI_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.INTERNAL",
    identifier: Some("RunMat:strcmpi:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "strcmpi: internal error",
};
pub const STRCMPI_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:strcmpi:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "strcmpi: too many output arguments",
};
pub const STRCMPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        STRCMPI_ERROR_INVALID_INPUT,
        STRCMPI_ERROR_SHAPE_MISMATCH,
        STRCMPI_ERROR_INTERNAL,
        STRCMPI_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const STRCMPI_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "strcmpi compares text containers. An unsupported direct integer or numeric input returns scalar logical false without provider access; malformed host cell elements are diagnosed by the container validator.",
};
