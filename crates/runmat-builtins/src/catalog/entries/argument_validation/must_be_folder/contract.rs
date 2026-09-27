use crate::*;
pub const MUST_BE_FOLDER_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeFolder is a text/path validator; integer host or resident values reject without numeric conversion or provider access." };
pub const MUST_BE_FOLDER_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FOLDER.OUTPUTS",
    identifier: Some("RunMat:mustBeFolder:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeFolder: too many output arguments",
};
pub const MUST_BE_FOLDER_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FOLDER.INVALID",
    identifier: Some("RunMat:mustBeFolder:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeFolder: invalid argument",
};
pub const MUST_BE_FOLDER_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FOLDER.FAILED",
    identifier: Some("RunMat:mustBeFolder:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeFolder: value does not satisfy validator",
};
pub const MUST_BE_FOLDER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeFolder(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_FOLDER_ERROR_OUTPUTS,
        MUST_BE_FOLDER_ERROR_INVALID,
        MUST_BE_FOLDER_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
    ],
};
