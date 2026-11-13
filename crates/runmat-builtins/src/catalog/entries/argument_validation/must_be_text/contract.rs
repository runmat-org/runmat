use crate::*;
pub const MUST_BE_TEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeText is a text validator; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const MUST_BE_TEXT_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeText:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeText: too many output arguments",
};
pub const MUST_BE_TEXT_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeText:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeText: invalid argument",
};
pub const MUST_BE_TEXT_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeText:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeText: value does not satisfy validator",
};
pub const MUST_BE_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeText(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_TEXT_ERROR_OUTPUTS,
        MUST_BE_TEXT_ERROR_INVALID,
        MUST_BE_TEXT_ERROR_FAILED,
    ],
};
