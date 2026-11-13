use crate::*;
pub const MUST_BE_SPARSE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
 form: "mustBeSparse(integer_A)", inputs: &super::super::support::VALIDATOR_INTEGER_VALUE,
 computation_domain: BuiltinIntegerComputationDomain::Predicate,
 output_class: BuiltinIntegerOutputClassRule::NotApplicable,
 overflow: BuiltinIntegerOverflowRule::NotApplicable,
 backend: BuiltinIntegerBackendRule::HostAndGpu, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Every empty integer array passes as documented; nonempty dense or resident integer arrays fail from storage metadata, while RunMat-native sparse integer values pass without materialization.",
}];
pub const MUST_BE_SPARSE_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SPARSE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:mustBeSparse:TooManyOutputs"),
    when: "Any output is explicitly requested from this zero-output validator.",
    message: "mustBeSparse: too many output arguments",
};
pub const MUST_BE_SPARSE_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SPARSE.INVALID_ARGUMENT",
    identifier: Some("RunMat:mustBeSparse:InvalidArgument"),
    when: "The supported input count, control type, or option contract is violated.",
    message: "mustBeSparse: invalid argument",
};
pub const MUST_BE_SPARSE_ERROR_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MUST_BE_SPARSE.VALIDATION_FAILED",
    identifier: Some("RunMat:mustBeSparse:ValidationFailed"),
    when: "The supplied value does not satisfy the validator.",
    message: "mustBeSparse: value does not satisfy validator",
};
pub const MUST_BE_SPARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "mustBeSparse(A)",
        inputs: &[super::super::support::VALUE],
        outputs: &[],
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MUST_BE_SPARSE_ERROR_OUTPUTS,
        MUST_BE_SPARSE_ERROR_INVALID,
        MUST_BE_SPARSE_ERROR_FAILED,
        super::super::support::VALIDATION_ERROR_PROVIDER_OWNERSHIP,
        super::super::support::VALIDATION_ERROR_PROVIDER_PAYLOAD,
    ],
};
