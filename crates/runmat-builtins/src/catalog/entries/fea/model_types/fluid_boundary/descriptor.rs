//! Signatures, parameters, and integer facts for `fea.FluidBoundary` constructors.
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

pub const FEA_FLUID_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
};

const OUTLET_PRESSURE_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "Static outlet pressure as fea.Pressure (Pa).",
    ),
];
const OUTLET_PRESSURE_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.FluidBoundary boundary object.",
)];
const OUTLET_PRESSURE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.FluidBoundary.outletPressure(id, region, pressure)",
    inputs: &OUTLET_PRESSURE_INPUTS,
    outputs: &OUTLET_PRESSURE_OUTPUTS,
}];
pub const FEA_FLUID_BOUNDARY_OUTLET_PRESSURE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &OUTLET_PRESSURE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
