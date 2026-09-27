use crate::*;
pub const MUST_BE_NONPOSITIVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "mustBeNonpositive(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "All elements compare exactly against zero.",
    }];
pub const MUST_BE_NONPOSITIVE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONPOSITIVE.OUTPUTS",
    identifier: Some("RunMat:mustBeNonpositive:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNonpositive: too many output arguments",
};
pub const MUST_BE_NONPOSITIVE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONPOSITIVE.INVALID",
    identifier: Some("RunMat:mustBeNonpositive:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNonpositive: invalid argument",
};
pub const MUST_BE_NONPOSITIVE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONPOSITIVE.FAILED",
    identifier: Some("RunMat:mustBeNonpositive:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNonpositive: value does not satisfy validator",
};
pub const MUST_BE_NONPOSITIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonpositive(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONPOSITIVE_ERROR_OUTPUTS,
        MUST_BE_NONPOSITIVE_ERROR_INVALID,
        MUST_BE_NONPOSITIVE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
