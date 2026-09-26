use crate::*;
pub const MUST_BE_NONSPARSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeNonsparse(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Dense host and resident integer arrays pass from storage metadata; sparse integer arrays fail.",
}];
pub const MUST_BE_NONSPARSE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONSPARSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeNonsparse:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeNonsparse: too many output arguments",
};
pub const MUST_BE_NONSPARSE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONSPARSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeNonsparse:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeNonsparse: invalid argument",
};
pub const MUST_BE_NONSPARSE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_NONSPARSE.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeNonsparse:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeNonsparse: value does not satisfy validator",
};
pub const MUST_BE_NONSPARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeNonsparse(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_NONSPARSE_ERROR_OUTPUTS,
        MUST_BE_NONSPARSE_ERROR_INVALID,
        MUST_BE_NONSPARSE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
