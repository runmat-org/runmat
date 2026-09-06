use crate::*;

const LISTING: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "list",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Platform-formatted character array containing the matching names.",
};
const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "File, folder, or wildcard name as a character vector or string scalar.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "list = ls()",
        inputs: &[],
        outputs: &[LISTING],
    },
    BuiltinSignatureDescriptor {
        label: "list = ls(name)",
        inputs: &[NAME],
        outputs: &[LISTING],
    },
];
pub const LS_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LS.ARITY",
    identifier: Some("RunMat:ls:InvalidArity"),
    when: "More than one input is supplied.",
    message: "ls: too many input arguments",
};
pub const LS_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LS.NAME",
    identifier: Some("RunMat:ls:InvalidName"),
    when: "The name input is not a character vector or string scalar.",
    message: "ls: name must be a character vector or string scalar",
};
pub const LS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[LS_ERROR_ARITY, LS_ERROR_NAME],
};
pub const LS_PORTABLE_ROWS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ls-portable-row-layout",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "RunMat mode returns one padded character row per entry on every platform",
    error_identifier: None,
};
pub const LS_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "File and folder names are host text. Numeric and provider-resident values reject before filesystem or provider access." };
