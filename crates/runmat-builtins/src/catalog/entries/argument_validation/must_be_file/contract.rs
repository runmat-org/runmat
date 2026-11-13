use crate::*;
pub const MUST_BE_FILE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeFile is a text/path validator; integer host or resident values reject without numeric conversion or provider access." };
pub const MUST_BE_FILE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FILE.OUTPUTS",
    identifier: Some("RunMat:mustBeFile:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeFile: too many output arguments",
};
pub const MUST_BE_FILE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FILE.INVALID",
    identifier: Some("RunMat:mustBeFile:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeFile: invalid argument",
};
pub const MUST_BE_FILE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FILE.FAILED",
    identifier: Some("RunMat:mustBeFile:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeFile: value does not satisfy validator",
};
pub const MUST_BE_FILE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeFile(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_FILE_ERROR_OUTPUTS,
        MUST_BE_FILE_ERROR_INVALID,
        MUST_BE_FILE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
    ],
};
