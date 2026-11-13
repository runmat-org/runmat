use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "text",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Literal text to match.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host pattern value.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pat = pattern(text)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const PATTERN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATTERN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pattern:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "pattern: too many output arguments",
};
pub const PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[PATTERN_ERROR_TOO_MANY_OUTPUTS],
};
pub const PATTERN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only host text is accepted. Integer, numeric and resident values reject before provider access; no implicit numeric-to-text conversion occurs.",
};
