use crate::*;
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when a handle object or listener remains valid.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Handle object or event listener.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isvalid(value)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
pub const ISVALID_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVALID.INVALID_INPUT",
    identifier: Some("RunMat:isvalid:InvalidInput"),
    when: "The input is not a handle object or listener.",
    message: "isvalid: input must be a handle object or listener",
};
pub const ISVALID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISVALID_ERROR_INVALID_INPUT],
};
pub const ISVALID_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Fundamental integer host or resident values are invalid-domain inputs and reject from class metadata before payload/provider access." };
