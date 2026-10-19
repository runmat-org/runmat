use crate::*;
pub const MUST_BE_MEMBER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeMember(integer_A, integer_or_double_S)", inputs: &super::super::support::VALIDATOR_INTEGER_MEMBER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Exact membership preserves the documented same-class rule and double cross-class exception without rounding wide integers." }];
pub const MUST_BE_MEMBER_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_MEMBER.OUTPUTS",
    identifier: Some("RunMat:mustBeMember:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeMember: too many output arguments",
};
pub const MUST_BE_MEMBER_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_MEMBER.INVALID",
    identifier: Some("RunMat:mustBeMember:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeMember: invalid argument",
};
pub const MUST_BE_MEMBER_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_MEMBER.FAILED",
    identifier: Some("RunMat:mustBeMember:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeMember: value does not satisfy validator",
};
pub const MUST_BE_MEMBER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeMember(A, bound)",
        inputs: &[
            super::super::support::VALUE,
            BuiltinParamDescriptor {
                name: "S",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Comparison or membership operand; exact storage is preserved.",
            },
        ],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_MEMBER_ERROR_OUTPUTS,
        MUST_BE_MEMBER_ERROR_INVALID,
        MUST_BE_MEMBER_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
