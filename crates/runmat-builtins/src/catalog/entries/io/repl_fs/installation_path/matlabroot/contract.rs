use crate::*;

const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "root",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat installation root as a character row.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "root = matlabroot()",
    inputs: &[],
    outputs: &[OUTPUT],
}];

pub const MATLABROOT_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MATLABROOT.TOO_MANY_INPUTS",
    identifier: Some("RunMat:matlabroot:TooManyInputs"),
    when: "Any input is supplied.",
    message: "matlabroot: too many input arguments",
};

pub const MATLABROOT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MATLABROOT_ERROR_TOO_MANY_INPUTS],
};

pub const MATLABROOT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "matlabroot accepts no inputs and returns text.",
};
