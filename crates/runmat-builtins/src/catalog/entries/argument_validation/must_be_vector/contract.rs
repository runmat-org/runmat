use crate::*;
pub const MUST_BE_VECTOR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "mustBeVector(integer_A)",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The documented 1-by-N or N-by-1 rule, including only vector-shaped empties, is decided from host or resident shape metadata without payload access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "mustBeVector(integer_A, \"allow-all-empties\")",
        inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility target's option additionally accepts every empty integer shape while preserving the ordinary vector rule for nonempty values; callable and arguments-block paths share the same metadata-only predicate.",
    },
];
pub const MUST_BE_VECTOR_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_VECTOR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeVector:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeVector: too many output arguments",
};
pub const MUST_BE_VECTOR_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_VECTOR.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeVector:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeVector: invalid argument",
};
pub const MUST_BE_VECTOR_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_VECTOR.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeVector:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeVector: value does not satisfy validator",
};
pub const MUST_BE_VECTOR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "mustBeVector(A)",
            inputs: &[super::super::support::VALUE],
            outputs: &[],
        },
        BuiltinSignatureDescriptor {
            label: "mustBeVector(A, 'allow-all-empties')",
            inputs: &[
                super::super::support::VALUE,
                BuiltinParamDescriptor {
                    name: "option",
                    ty: BuiltinParamType::Any,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: "The case-insensitive allow-all-empties flag.",
                },
            ],
            outputs: &[],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_VECTOR_ERROR_OUTPUTS,
        MUST_BE_VECTOR_ERROR_INVALID,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        MUST_BE_VECTOR_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
