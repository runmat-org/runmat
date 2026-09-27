use crate::*;
pub const MUST_BE_REAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeReal(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "All eight native integer classes are real by construction; host and resident values pass from authoritative class/storage metadata without floating conversion or payload access." }];
pub const MUST_BE_REAL_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_REAL.OUTPUTS",
    identifier: Some("RunMat:mustBeReal:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeReal: too many output arguments",
};
pub const MUST_BE_REAL_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_REAL.INVALID",
    identifier: Some("RunMat:mustBeReal:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeReal: invalid argument",
};
pub const MUST_BE_REAL_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_REAL.FAILED",
    identifier: Some("RunMat:mustBeReal:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeReal: value does not satisfy validator",
};
pub const MUST_BE_REAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeReal(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_REAL_ERROR_OUTPUTS,
        MUST_BE_REAL_ERROR_INVALID,
        MUST_BE_REAL_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
