use crate::*;
pub const MUST_BE_GREATER_THAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "mustBeGreaterThan(integer_A, numeric_B)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE_AND_BOUND, computation_domain: BuiltinIntegerComputationDomain::Predicate, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::BroadcastCompatible, notes: "The compatibility target's compatible-size comparison is exact across mixed integer and floating classes; resident values use an owner-preserving predicate path." }];
pub const MUST_BE_GREATER_THAN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_GREATER_THAN.OUTPUTS",
    identifier: Some("RunMat:mustBeGreaterThan:TooManyOutputs"),
    when: "An output is explicitly requested.",
    message: "mustBeGreaterThan: too many output arguments",
};
pub const MUST_BE_GREATER_THAN_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_GREATER_THAN.INVALID",
    identifier: Some("RunMat:mustBeGreaterThan:InvalidArgument"),
    when: "Input count, class, shape or control admission fails.",
    message: "mustBeGreaterThan: invalid argument",
};
pub const MUST_BE_GREATER_THAN_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_GREATER_THAN.FAILED",
    identifier: Some("RunMat:mustBeGreaterThan:ValidationFailed"),
    when: "The supplied value does not satisfy this validator.",
    message: "mustBeGreaterThan: value does not satisfy validator",
};
pub const MUST_BE_GREATER_THAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeGreaterThan(A, bound)",
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
        MUST_BE_GREATER_THAN_ERROR_OUTPUTS,
        MUST_BE_GREATER_THAN_ERROR_INVALID,
        MUST_BE_GREATER_THAN_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD,
        super::super::support::VALIDATION_ERROR_DOWNLOAD_PAYLOAD,
    ],
};
