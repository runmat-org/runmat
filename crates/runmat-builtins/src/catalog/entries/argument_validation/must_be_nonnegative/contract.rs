use crate::*;
pub const MUST_BE_NONNEGATIVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNonnegative(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All elements compare exactly against zero.",
    }];
pub const MUST_BE_NONNEGATIVE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONNEGATIVE.OUTPUTS",
    identifier: Some("RunMat:mustBeNonnegative:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNonnegative: too many output arguments",
};
pub const MUST_BE_NONNEGATIVE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONNEGATIVE.INVALID",
    identifier: Some("RunMat:mustBeNonnegative:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNonnegative: invalid argument",
};
pub const MUST_BE_NONNEGATIVE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONNEGATIVE.FAILED",
    identifier: Some("RunMat:mustBeNonnegative:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNonnegative: value does not satisfy validator",
};
pub const MUST_BE_NONNEGATIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonnegative(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONNEGATIVE_ERROR_OUTPUTS,
        MUST_BE_NONNEGATIVE_ERROR_INVALID,
        MUST_BE_NONNEGATIVE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
