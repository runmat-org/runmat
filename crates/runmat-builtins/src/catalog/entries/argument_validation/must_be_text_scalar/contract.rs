use crate::*;
pub const MUST_BE_TEXT_SCALAR_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeTextScalar is a text-shape validator; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const MUST_BE_TEXT_SCALAR_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT_SCALAR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeTextScalar:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeTextScalar: too many output arguments",
};
pub const MUST_BE_TEXT_SCALAR_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT_SCALAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeTextScalar:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeTextScalar: invalid argument",
};
pub const MUST_BE_TEXT_SCALAR_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_TEXT_SCALAR.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeTextScalar:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeTextScalar: value does not satisfy validator",
};
pub const MUST_BE_TEXT_SCALAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeTextScalar(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_TEXT_SCALAR_ERROR_OUTPUTS,
        MUST_BE_TEXT_SCALAR_ERROR_INVALID,
        MUST_BE_TEXT_SCALAR_ERROR_FAILED,
    ],
};
