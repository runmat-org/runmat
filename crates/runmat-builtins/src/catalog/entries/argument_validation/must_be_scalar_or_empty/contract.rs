use crate::*;
pub const MUST_BE_SCALAR_OR_EMPTY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeScalarOrEmpty(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Scalar-or-empty validation reads shape metadata only, including all documented empty integer shapes, without gathering resident payloads.",
}];
pub const MUST_BE_SCALAR_OR_EMPTY_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SCALAR_OR_EMPTY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeScalarOrEmpty:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeScalarOrEmpty: too many output arguments",
};
pub const MUST_BE_SCALAR_OR_EMPTY_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SCALAR_OR_EMPTY.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeScalarOrEmpty:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeScalarOrEmpty: invalid argument",
};
pub const MUST_BE_SCALAR_OR_EMPTY_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SCALAR_OR_EMPTY.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeScalarOrEmpty:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeScalarOrEmpty: value does not satisfy validator",
};
pub const MUST_BE_SCALAR_OR_EMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeScalarOrEmpty(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_SCALAR_OR_EMPTY_ERROR_OUTPUTS,
        MUST_BE_SCALAR_OR_EMPTY_ERROR_INVALID,
        MUST_BE_SCALAR_OR_EMPTY_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
