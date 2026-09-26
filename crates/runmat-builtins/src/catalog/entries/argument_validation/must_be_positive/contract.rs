use crate::*;
pub const MUST_BE_POSITIVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBePositive(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All elements compare exactly against zero; empty arrays pass.",
    }];
pub const MUST_BE_POSITIVE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_POSITIVE.OUTPUTS",
    identifier: Some("RunMat:mustBePositive:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBePositive: too many output arguments",
};
pub const MUST_BE_POSITIVE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_POSITIVE.INVALID",
    identifier: Some("RunMat:mustBePositive:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBePositive: invalid argument",
};
pub const MUST_BE_POSITIVE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_POSITIVE.FAILED",
    identifier: Some("RunMat:mustBePositive:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBePositive: value does not satisfy validator",
};
pub const MUST_BE_POSITIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBePositive(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_POSITIVE_ERROR_OUTPUTS,
        MUST_BE_POSITIVE_ERROR_INVALID,
        MUST_BE_POSITIVE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
