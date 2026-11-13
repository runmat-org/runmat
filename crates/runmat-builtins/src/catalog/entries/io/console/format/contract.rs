use crate::*;

const MODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "mode",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Case-insensitive numeric mode or accepted compact/loose spacing keyword.",
};
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "format()",
        inputs: &[],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "format(mode)",
        inputs: &[MODE],
        outputs: &[],
    },
];
pub const FORMAT_ERROR_ARG_CONFIG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FORMAT.ARG_CONFIG",
    identifier: None,
    when: "Inputs do not match a zero- or one-text-argument call form.",
    message: "format: invalid argument configuration",
};
pub const FORMAT_ERROR_UNKNOWN_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FORMAT.UNKNOWN_MODE",
    identifier: None,
    when: "A text argument is not a supported numeric or spacing keyword.",
    message: "format: unknown format mode",
};
pub const FORMAT_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FORMAT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:format:TooManyOutputs"),
    when: "One or more output arguments are requested from format.",
    message: "format: expected no output arguments",
};
pub const FORMAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        FORMAT_ERROR_ARG_CONFIG,
        FORMAT_ERROR_UNKNOWN_MODE,
        FORMAT_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const FORMAT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "format accepts a text mode or no input; it changes display state but does not convert stored numeric or integer values.",
};
