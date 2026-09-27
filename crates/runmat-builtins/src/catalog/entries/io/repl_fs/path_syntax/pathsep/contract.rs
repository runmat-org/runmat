use crate::*;

pub const PATHSEP_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATHSEP.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pathsep:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "pathsep: too many output arguments",
};
const OUTPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "separator",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform search-path-list separator character.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "separator = pathsep()",
    inputs: &[],
    outputs: &[OUTPUT],
}];
pub const PATHSEP_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PATHSEP.ARITY",
    identifier: Some("RunMat:pathsep:TooManyInputs"),
    when: "Any input is supplied.",
    message: "pathsep: too many input arguments",
};
pub const PATHSEP_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[PATHSEP_ERROR_ARITY, PATHSEP_ERROR_TOO_MANY_OUTPUTS],
};
pub const PATHSEP_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "pathsep accepts no inputs and returns one character.",
};
