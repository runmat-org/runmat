use crate::*;
pub const MUST_BE_A_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeA(integer_A, class_names)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Class validation uses wrapper identity for explicit gpuArray values and the underlying integer class for ordinary host or internally automatic-resident values; payload data is never converted.",
}];
pub const MUST_BE_A_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_A.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeA:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeA: too many output arguments",
};
pub const MUST_BE_A_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_A.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeA:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeA: invalid argument",
};
pub const MUST_BE_A_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_A.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeA:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeA: value does not satisfy validator",
};
pub const MUST_BE_A_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeA(A, classes)",
        inputs: &[
            super::super::support::VALUE,
            super::super::support::CLASS_NAMES,
        ],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_A_ERROR_OUTPUTS,
        MUST_BE_A_ERROR_INVALID,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        MUST_BE_A_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
