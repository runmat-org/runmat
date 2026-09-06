use crate::*;
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "separator",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform file-name separator character.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "separator = filesep()",
    inputs: &[],
    outputs: &[OUTPUT],
}];
pub const FILESEP_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILESEP.ARITY",
    identifier: Some("RunMat:filesep:TooManyInputs"),
    when: "Any input is supplied.",
    message: "filesep: too many input arguments",
};
pub const FILESEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[FILESEP_ERROR_ARITY],
};
pub const FILESEP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "filesep accepts no inputs and returns one character.",
};
