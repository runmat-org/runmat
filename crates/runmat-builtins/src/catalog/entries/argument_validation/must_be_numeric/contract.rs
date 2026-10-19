use crate::*;
pub const MUST_BE_NUMERIC_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNumeric(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All native integer classes satisfy isnumeric-style validation from class metadata.",
    }];
pub const MUST_BE_NUMERIC_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NUMERIC.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeNumeric:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeNumeric: too many output arguments",
};
pub const MUST_BE_NUMERIC_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NUMERIC.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeNumeric:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeNumeric: invalid argument",
};
pub const MUST_BE_NUMERIC_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NUMERIC.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeNumeric:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeNumeric: value does not satisfy validator",
};
pub const MUST_BE_NUMERIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNumeric(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NUMERIC_ERROR_OUTPUTS,
        MUST_BE_NUMERIC_ERROR_INVALID,
        MUST_BE_NUMERIC_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
