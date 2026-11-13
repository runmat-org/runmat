//! Signatures, parameters, and integer facts for `fea.StructuralBoundary` constructors.
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

pub const FEA_STRUCTURAL_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
};

const PRESSURE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
    required(
        "pressure",
        BuiltinParamType::Any,
        "Gauge pressure as fea.Pressure (Pa).",
    ),
];
const PRESSURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.StructuralBoundary boundary object.",
)];
const PRESSURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.StructuralBoundary.pressure(id, region, pressure)",
    inputs: &PRESSURE_INPUTS,
    outputs: &PRESSURE_OUTPUTS,
}];
pub const FEA_STRUCTURAL_BOUNDARY_PRESSURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PRESSURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
