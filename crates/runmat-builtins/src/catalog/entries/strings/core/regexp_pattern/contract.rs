use crate::*;
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "expr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Regular-expression text.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "pat",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host pattern value.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pat = regexpPattern(expr)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.REGEXP_PATTERN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:regexpPattern:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "regexpPattern: too many output arguments",
};
pub const REGEXP_PATTERN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[REGEXP_PATTERN_ERROR_TOO_MANY_OUTPUTS],
};
pub const REGEXP_PATTERN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only host regular-expression text is accepted. Integer, numeric and resident values reject before provider access.",
};
