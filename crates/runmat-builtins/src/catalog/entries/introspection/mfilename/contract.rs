use crate::*;

const NAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host string scalar naming the lexical source or method class.",
}];
const OPTION: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "option",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "'fullpath' or 'class'; other values preserve the default name mode.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "name = mfilename()",
        inputs: &[],
        outputs: &NAME,
    },
    BuiltinSignatureDescriptor {
        label: "name = mfilename(option)",
        inputs: &OPTION,
        outputs: &NAME,
    },
];

pub const MFILENAME_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MFILENAME.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "More than one option argument is provided.",
    message: "mfilename: too many input arguments",
};
pub const MFILENAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MFILENAME_ERROR_TOO_MANY_INPUTS],
};
pub const MFILENAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None,
    notes: "An integer option is not a source mode and retains the runtime's default-name behavior; no numeric data are inspected or converted.",
};
