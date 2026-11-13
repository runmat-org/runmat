use crate::*;
pub const MUST_BE_COLUMN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeColumn(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Column validation reads shape metadata only and supports resident integer arrays without gathering.",
}];
pub const MUST_BE_COLUMN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_COLUMN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeColumn:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeColumn: too many output arguments",
};
pub const MUST_BE_COLUMN_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_COLUMN.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeColumn:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeColumn: invalid argument",
};
pub const MUST_BE_COLUMN_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_COLUMN.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeColumn:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeColumn: value does not satisfy validator",
};
pub const MUST_BE_COLUMN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeColumn(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_COLUMN_ERROR_OUTPUTS,
        MUST_BE_COLUMN_ERROR_INVALID,
        MUST_BE_COLUMN_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
