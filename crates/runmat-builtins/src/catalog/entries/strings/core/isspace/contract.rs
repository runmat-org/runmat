use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input text.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical classification result.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isspace(text)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISSPACE_ERROR_INVALID_STRING_ARRAY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSPACE.INVALID_STRING_ARRAY",
    identifier: Some("RunMat:isspace:InvalidStringArray"),
    when: "A string array contains other than one element.",
    message: "isspace: string input must be a string scalar",
};
pub const ISSPACE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISSPACE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:isspace:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "isspace: too many output arguments",
};
pub const ISSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        ISSPACE_ERROR_INVALID_STRING_ARRAY,
        ISSPACE_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const ISSPACE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only character arrays and string scalars are classified. Integer host or resident values return scalar false without reading numeric payloads or contacting a provider.",
};
