use crate::*;
pub const ISVARNAME_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor =
    BuiltinIntegerAuditDescriptor {
        kind: BuiltinIntegerAuditKind::NotApplicable,
        canonical_builtin: None,
        notes: "isvarname is a text predicate; integer host or resident values return scalar false without numeric conversion or provider access.",
    };
pub const ISVARNAME_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVARNAME.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:isvarname:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "isvarname: too many output arguments",
};
pub const ISVARNAME_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVARNAME.INVALID_ARGUMENT",
    identifier: Some("RunMat:isvarname:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "isvarname: invalid argument",
};
pub const ISVARNAME_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISVARNAME.VALIDATION_FAILED",
    identifier: Some("RunMat:isvarname:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "isvarname: value does not satisfy validator",
};
pub const ISVARNAME_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "isvarname(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[super::super::support::LOGICAL_OUTPUT],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[ISVARNAME_ERROR_OUTPUTS, ISVARNAME_ERROR_INVALID],
};
