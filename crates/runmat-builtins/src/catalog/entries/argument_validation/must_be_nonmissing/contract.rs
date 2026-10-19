use crate::*;
pub const MUST_BE_NONMISSING_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeNonmissing(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "Native integer classes have no standard missing representation and therefore pass without conversion; public validator and delegated anymissing type lists are not fully synchronized." }];
pub const MUST_BE_NONMISSING_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONMISSING.OUTPUTS",
    identifier: Some("RunMat:mustBeNonmissing:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeNonmissing: too many output arguments",
};
pub const MUST_BE_NONMISSING_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONMISSING.INVALID",
    identifier: Some("RunMat:mustBeNonmissing:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeNonmissing: invalid argument",
};
pub const MUST_BE_NONMISSING_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONMISSING.FAILED",
    identifier: Some("RunMat:mustBeNonmissing:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeNonmissing: value does not satisfy validator",
};
pub const MUST_BE_NONMISSING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonmissing(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONMISSING_ERROR_OUTPUTS,
        MUST_BE_NONMISSING_ERROR_INVALID,
        MUST_BE_NONMISSING_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
