use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, OUTPUT};
use crate::*;
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "C = mpower(A, B)",
    inputs: &[INPUT_A, INPUT_B],
    outputs: &[OUTPUT],
}];
pub const MPOWER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MPOWER.INVALID_ARGUMENT",
    identifier: Some("RunMat:mpower:InvalidArgument"),
    when: "The exponent is not a supported scalar value.",
    message: "mpower: invalid exponent",
};
pub const MPOWER_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MPOWER.INVALID_INPUT",
    identifier: Some("RunMat:mpower:InvalidInput"),
    when: "The base is unsupported or a matrix base is not square.",
    message: "mpower: unsupported operand types",
};
pub const MPOWER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MPOWER.INTERNAL",
    identifier: Some("RunMat:mpower:Internal"),
    when: "Runtime cannot complete a provider operation or materialize its output.",
    message: "mpower: internal runtime failure",
};
pub const MPOWER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        MPOWER_ERROR_INVALID_ARGUMENT,
        MPOWER_ERROR_INVALID_INPUT,
        MPOWER_ERROR_INTERNAL,
    ],
};
pub const MPOWER_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "C = mpower(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Scalar integer powers and square integer matrices raised to a nonnegative integer-valued scalar preserve the base class. Matrix products use saturating multiply-accumulate operations; resident integer inputs gather exactly and return to the original owner." }];
