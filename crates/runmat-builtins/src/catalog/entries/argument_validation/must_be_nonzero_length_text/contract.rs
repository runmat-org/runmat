use crate::*;
pub const MUST_BE_NONZERO_LENGTH_TEXT_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeNonzeroLengthText is a text validator; integer host or resident values fail without numeric conversion or provider access." };
pub const MUST_BE_NONZERO_LENGTH_TEXT_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NONZERO_LENGTH_TEXT.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:mustBeNonzeroLengthText:TooManyOutputs"),
        when: "Any output is explicitly requested from this zero-output validator.",
        message: "mustBeNonzeroLengthText: too many output arguments",
    };
pub const MUST_BE_NONZERO_LENGTH_TEXT_ERROR_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NONZERO_LENGTH_TEXT.INVALID_ARGUMENT",
        identifier: Some("RunMat:mustBeNonzeroLengthText:InvalidArgument"),
        when: "The supported input count, control type, or option contract is violated.",
        message: "mustBeNonzeroLengthText: invalid argument",
    };
pub const MUST_BE_NONZERO_LENGTH_TEXT_ERROR_FAILED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NONZERO_LENGTH_TEXT.VALIDATION_FAILED",
        identifier: Some("RunMat:mustBeNonzeroLengthText:ValidationFailed"),
        when: "The supplied value does not satisfy the validator.",
        message: "mustBeNonzeroLengthText: value does not satisfy validator",
    };
pub const MUST_BE_NONZERO_LENGTH_TEXT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonzeroLengthText(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONZERO_LENGTH_TEXT_ERROR_OUTPUTS,
        MUST_BE_NONZERO_LENGTH_TEXT_ERROR_INVALID,
        MUST_BE_NONZERO_LENGTH_TEXT_ERROR_FAILED,
    ],
};
