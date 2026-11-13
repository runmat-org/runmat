use crate::*;
pub const MUST_BE_FLOAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeFloat(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::FunctionSpecific, notes: "Every native integer class fails isfloat-style validation from metadata without gathering or conversion.",
}];
pub const MUST_BE_FLOAT_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FLOAT.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeFloat:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeFloat: too many output arguments",
};
pub const MUST_BE_FLOAT_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FLOAT.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeFloat:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeFloat: invalid argument",
};
pub const MUST_BE_FLOAT_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_FLOAT.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeFloat:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeFloat: value does not satisfy validator",
};
pub const MUST_BE_FLOAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeFloat(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_FLOAT_ERROR_OUTPUTS,
        MUST_BE_FLOAT_ERROR_INVALID,
        MUST_BE_FLOAT_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
