use crate::*;

const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "folder",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "System temporary directory as a character row with a trailing separator.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "folder = tempdir()",
    inputs: &[],
    outputs: &[OUTPUT],
}];

pub const TEMPDIR_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPDIR.TOO_MANY_INPUTS",
    identifier: Some("RunMat:tempdir:TooManyInputs"),
    when: "Any input is supplied.",
    message: "tempdir: too many input arguments",
};
pub const TEMPDIR_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TEMPDIR.UNAVAILABLE",
    identifier: Some("RunMat:tempdir:Unavailable"),
    when: "The session cannot determine a temporary directory.",
    message: "tempdir: unable to determine temporary directory",
};

pub const TEMPDIR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[TEMPDIR_ERROR_TOO_MANY_INPUTS, TEMPDIR_ERROR_UNAVAILABLE],
};

pub const TEMPDIR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "tempdir accepts no inputs and returns text.",
};
