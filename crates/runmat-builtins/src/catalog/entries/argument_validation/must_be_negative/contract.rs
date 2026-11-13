use crate::*;
pub const MUST_BE_NEGATIVE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeNegative(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "All elements compare exactly against zero; unsigned nonempty arrays fail and empty arrays pass." }];
pub const MUST_BE_NEGATIVE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NEGATIVE.OUTPUTS",
    identifier: Some("RunMat:mustBeNegative:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNegative: too many output arguments",
};
pub const MUST_BE_NEGATIVE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NEGATIVE.INVALID",
    identifier: Some("RunMat:mustBeNegative:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNegative: invalid argument",
};
pub const MUST_BE_NEGATIVE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NEGATIVE.FAILED",
    identifier: Some("RunMat:mustBeNegative:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNegative: value does not satisfy validator",
};
pub const MUST_BE_NEGATIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNegative(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NEGATIVE_ERROR_OUTPUTS,
        MUST_BE_NEGATIVE_ERROR_INVALID,
        MUST_BE_NEGATIVE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
