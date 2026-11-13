use crate::*;
pub const MUST_BE_NONEMPTY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNonempty(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Emptiness is decided from shape metadata without payload access.",
    }];
pub const MUST_BE_NONEMPTY_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONEMPTY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeNonempty:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeNonempty: too many output arguments",
};
pub const MUST_BE_NONEMPTY_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONEMPTY.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeNonempty:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeNonempty: invalid argument",
};
pub const MUST_BE_NONEMPTY_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONEMPTY.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeNonempty:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeNonempty: value does not satisfy validator",
};
pub const MUST_BE_NONEMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonempty(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONEMPTY_ERROR_OUTPUTS,
        MUST_BE_NONEMPTY_ERROR_INVALID,
        MUST_BE_NONEMPTY_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
