use crate::*;
pub const MUST_BE_VALID_VARIABLE_NAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "mustBeValidVariableName accepts text names only; integer host or resident values fail before numeric conversion, payload access, or provider lookup." };
pub const MUST_BE_VALID_VARIABLE_NAME_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_VALID_VARIABLE_NAME.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:mustBeValidVariableName:TooManyOutputs"),
        when: "Any output is explicitly requested from this zero-output validator.",
        message: "mustBeValidVariableName: too many output arguments",
    };
pub const MUST_BE_VALID_VARIABLE_NAME_ERROR_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_VALID_VARIABLE_NAME.INVALID_ARGUMENT",
        identifier: Some("RunMat:mustBeValidVariableName:InvalidArgument"),
        when: "The supported input count, control type, or option contract is violated.",
        message: "mustBeValidVariableName: invalid argument",
    };
pub const MUST_BE_VALID_VARIABLE_NAME_ERROR_FAILED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_VALID_VARIABLE_NAME.VALIDATION_FAILED",
        identifier: Some("RunMat:mustBeValidVariableName:ValidationFailed"),
        when: "The supplied value does not satisfy the validator.",
        message: "mustBeValidVariableName: value does not satisfy validator",
    };
pub const MUST_BE_VALID_VARIABLE_NAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeValidVariableName(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_VALID_VARIABLE_NAME_ERROR_OUTPUTS,
        MUST_BE_VALID_VARIABLE_NAME_ERROR_INVALID,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        MUST_BE_VALID_VARIABLE_NAME_ERROR_FAILED,
    ],
};
