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
    label: "tf = isletter(text)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISLETTER_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISLETTER.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:isletter:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "isletter: too many output arguments",
};
pub const ISLETTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISLETTER_ERROR_TOO_MANY_OUTPUTS],
};
pub const ISLETTER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "Only host string scalars and character arrays are classified. Integer and resident numeric inputs return scalar false without reading numeric payloads or contacting a provider.",
};
