use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, OUTPUT};
use crate::*;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "X = mldivide(A, B)",
    inputs: &[INPUT_A, INPUT_B],
    outputs: &[OUTPUT],
}];

pub const MLDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MLDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:mldivide:InvalidInput"),
    when: "Inputs are unsupported, are not matrices, or have incompatible dimensions.",
    message: "mldivide: unsupported operand types",
};
pub const MLDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MLDIVIDE.INTERNAL",
    identifier: Some("RunMat:mldivide:Internal"),
    when: "The solver or acceleration provider cannot complete the operation correctly.",
    message: "mldivide: internal runtime failure",
};
pub const MLDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MLDIVIDE_ERROR_INVALID_INPUT, MLDIVIDE_ERROR_INTERNAL],
};
pub const MLDIVIDE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "X = mldivide(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Integer participation is limited to scalar left division. The operation is exact class-preserving element-wise division; integer matrix solves are rejected." }];
