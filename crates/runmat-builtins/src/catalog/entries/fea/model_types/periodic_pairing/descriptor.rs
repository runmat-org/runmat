//! Signatures, parameters, and integer facts for `fea.PeriodicPairing` constructors.
//! The runtime binding re-exports these facts; it does not redefine them.

use crate::fea_contract::descriptors::ERRORS;
use crate::*;

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

pub const FEA_PERIODIC_PAIRING_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Concrete constraint constructors accept identifiers, immutable geometry regions, and already-dimensioned values; they have no direct integer input role.",
};

const MATCHING_INPUTS: [BuiltinParamDescriptor; 0] = [];
const MATCHING_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "pairing",
    BuiltinParamType::Any,
    "fea.MatchingPeriodicPairing object.",
)];
const MATCHING_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pairing = fea.PeriodicPairing.matching()",
    inputs: &MATCHING_INPUTS,
    outputs: &MATCHING_OUTPUTS,
}];
pub const FEA_PERIODIC_PAIRING_MATCHING_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MATCHING_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const GEOMETRIC_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "tolerance",
    BuiltinParamType::Any,
    "Matching tolerance as nonnegative fea.Length (m).",
)];
const GEOMETRIC_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "pairing",
    BuiltinParamType::Any,
    "fea.GeometricPeriodicPairing object.",
)];
const GEOMETRIC_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "pairing = fea.PeriodicPairing.geometric(tolerance)",
    inputs: &GEOMETRIC_INPUTS,
    outputs: &GEOMETRIC_OUTPUTS,
}];
pub const FEA_PERIODIC_PAIRING_GEOMETRIC_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GEOMETRIC_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
