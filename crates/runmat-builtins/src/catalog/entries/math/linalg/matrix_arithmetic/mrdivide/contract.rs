use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, OUTPUT};
use crate::*;
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "X = mrdivide(A, B)",
    inputs: &[INPUT_A, INPUT_B],
    outputs: &[OUTPUT],
}];
pub const MRDIVIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MRDIVIDE.INVALID_INPUT",
    identifier: Some("RunMat:mrdivide:InvalidInput"),
    when: "Inputs are unsupported, are not matrices, or have incompatible dimensions.",
    message: "mrdivide: unsupported operand types",
};
pub const MRDIVIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MRDIVIDE.INTERNAL",
    identifier: Some("RunMat:mrdivide:Internal"),
    when: "The solver or acceleration provider cannot complete the operation correctly.",
    message: "mrdivide: internal runtime failure",
};
pub const MRDIVIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MRDIVIDE_ERROR_INVALID_INPUT, MRDIVIDE_ERROR_INTERNAL],
};
pub const MRDIVIDE_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] = &[BuiltinIntegerCapabilityDescriptor { form: "X = mrdivide(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "Integer participation is limited to scalar right division. The operation is exact class-preserving element-wise division; integer matrix solves are rejected." }];
