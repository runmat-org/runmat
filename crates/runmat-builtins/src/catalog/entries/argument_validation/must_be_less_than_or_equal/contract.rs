use crate::*;
pub const MUST_BE_LESS_THAN_OR_EQUAL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeLessThanOrEqual(integer_A, numeric_B)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE_AND_BOUND, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes." }];
pub const MUST_BE_LESS_THAN_OR_EQUAL_ERROR_OUTPUTS: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_LESS_THAN_OR_EQUAL.OUTPUTS",
        identifier: Some("RunMat:mustBeLessThanOrEqual:TooManyOutputs"),
        when: "An output is explicitly requested.",
        message: "mustBeLessThanOrEqual: too many output arguments",
    };
pub const MUST_BE_LESS_THAN_OR_EQUAL_ERROR_INVALID: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_LESS_THAN_OR_EQUAL.INVALID",
        identifier: Some("RunMat:mustBeLessThanOrEqual:InvalidArgument"),
        when: "Input count, class, shape or control admission fails.",
        message: "mustBeLessThanOrEqual: invalid argument",
    };
pub const MUST_BE_LESS_THAN_OR_EQUAL_ERROR_FAILED: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.MUST_BE_LESS_THAN_OR_EQUAL.FAILED",
        identifier: Some("RunMat:mustBeLessThanOrEqual:ValidationFailed"),
        when: "The supplied value does not satisfy this validator.",
        message: "mustBeLessThanOrEqual: value does not satisfy validator",
    };
pub const MUST_BE_LESS_THAN_OR_EQUAL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeLessThanOrEqual(A, bound)",
        inputs: &[
            super::super::support::VALUE,
            BuiltinParamDescriptor {
                name: "bound",
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
        MUST_BE_LESS_THAN_OR_EQUAL_ERROR_OUTPUTS,
        MUST_BE_LESS_THAN_OR_EQUAL_ERROR_INVALID,
        MUST_BE_LESS_THAN_OR_EQUAL_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
