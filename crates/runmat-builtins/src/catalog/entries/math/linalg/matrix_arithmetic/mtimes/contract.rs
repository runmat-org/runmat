use super::super::contract::{INPUT_A, INPUT_B, INTEGER_INPUTS, OUTPUT};
use crate::*;

const SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "C = mtimes(A, B)",
    inputs: &[INPUT_A, INPUT_B],
    outputs: &[OUTPUT],
}];

pub const MTIMES_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MTIMES.INVALID_INPUT",
    identifier: Some("RunMat:mtimes:InvalidInput"),
    when: "Operands are unsupported or matrix dimensions are incompatible.",
    message: "mtimes: unsupported operand types",
};
pub const MTIMES_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MTIMES.INTERNAL",
    identifier: Some("RunMat:mtimes:Internal"),
    when: "Runtime cannot complete a provider operation or materialize its output.",
    message: "mtimes: internal runtime failure",
};

pub const MTIMES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[MTIMES_ERROR_INVALID_INPUT, MTIMES_ERROR_INTERNAL],
};

pub const MTIMES_INTEGER_CAPABILITIES: &[BuiltinIntegerCapabilityDescriptor] =
    &[BuiltinIntegerCapabilityDescriptor { form: "C = mtimes(A, B)", inputs: INTEGER_INPUTS, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput, overflow: BuiltinIntegerOverflowRule::Saturate, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "When either operand is integer, the other must be scalar and use the same integer class or scalar double. The supported scalar product preserves the integer class and saturates; resident fallback gathers authoritative typed storage and restores the original owner." }];
