use crate::*;
pub const MUST_BE_UNDERLYING_TYPE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeUnderlyingType(integer_A, typenames)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "One or more requested type names are compared with the authoritative signedness and width of host or resident integer storage without reading payload values.",
}];
pub const MUST_BE_UNDERLYING_TYPE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_UNDERLYING_TYPE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeUnderlyingType:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeUnderlyingType: too many output arguments",
};
pub const MUST_BE_UNDERLYING_TYPE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_UNDERLYING_TYPE.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeUnderlyingType:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeUnderlyingType: invalid argument",
};
pub const MUST_BE_UNDERLYING_TYPE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_UNDERLYING_TYPE.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeUnderlyingType:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeUnderlyingType: value does not satisfy validator",
};
pub const MUST_BE_UNDERLYING_TYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeUnderlyingType(A, classes)",
        inputs: &[
            super::super::support::VALUE,
            super::super::support::CLASS_NAMES,
        ],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_UNDERLYING_TYPE_ERROR_OUTPUTS,
        MUST_BE_UNDERLYING_TYPE_ERROR_INVALID,
        super::super::support::VALIDATION_ERROR_TEXT_INPUT,
        MUST_BE_UNDERLYING_TYPE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
