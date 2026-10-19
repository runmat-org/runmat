use crate::*;
pub const MUST_BE_NUMERIC_OR_LOGICAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNumericOrLogical(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "All native integer classes satisfy the numeric branch from class metadata.",
    }];
pub const MUST_BE_NUMERIC_OR_LOGICAL_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NUMERIC_OR_LOGICAL.TOO_MANY_OUTPUTS",
        identifier: Some("RunMat:mustBeNumericOrLogical:TooManyOutputs"),
        when: "Any output is explicitly requested from this zero-output validator.",
        message: "mustBeNumericOrLogical: too many output arguments",
    };
pub const MUST_BE_NUMERIC_OR_LOGICAL_ERROR_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NUMERIC_OR_LOGICAL.INVALID_ARGUMENT",
        identifier: Some("RunMat:mustBeNumericOrLogical:InvalidArgument"),
        when: "The supported input count, control type, or option contract is violated.",
        message: "mustBeNumericOrLogical: invalid argument",
    };
pub const MUST_BE_NUMERIC_OR_LOGICAL_ERROR_FAILED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_NUMERIC_OR_LOGICAL.VALIDATION_FAILED",
        identifier: Some("RunMat:mustBeNumericOrLogical:ValidationFailed"),
        when: "The supplied value does not satisfy the validator.",
        message: "mustBeNumericOrLogical: value does not satisfy validator",
    };
pub const MUST_BE_NUMERIC_OR_LOGICAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNumericOrLogical(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NUMERIC_OR_LOGICAL_ERROR_OUTPUTS,
        MUST_BE_NUMERIC_OR_LOGICAL_ERROR_INVALID,
        MUST_BE_NUMERIC_OR_LOGICAL_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
