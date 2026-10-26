//! Signatures, parameters, and integer facts for `fea.MagneticBoundary` constructors.
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

pub const FEA_MAGNETIC_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
};

const INSULATION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Boundary identifier (nonempty text).",
    ),
    required(
        "region",
        BuiltinParamType::Any,
        "geometry.Region selected from the model geometry.",
    ),
];
const INSULATION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.MagneticBoundary boundary object.",
)];
const INSULATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.MagneticBoundary.insulation(id, region)",
    inputs: &INSULATION_INPUTS,
    outputs: &INSULATION_OUTPUTS,
}];
pub const FEA_MAGNETIC_BOUNDARY_INSULATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INSULATION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
