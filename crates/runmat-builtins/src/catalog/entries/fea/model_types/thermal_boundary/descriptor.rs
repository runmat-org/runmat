//! Signatures, parameters, and integer facts for `fea.ThermalBoundary` constructors.
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

pub const FEA_THERMAL_BOUNDARY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA boundary constructors accept identifiers, immutable geometry.Region values, and typed physical quantities; they have no integer input role.",
};

const CONVECTION_INPUTS: [BuiltinParamDescriptor; 4] = [
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
        "ambientTemperature",
        BuiltinParamType::Any,
        "Absolute ambient temperature as fea.Temperature.",
    ),
    required(
        "coefficient",
        BuiltinParamType::Any,
        "Film coefficient as fea.HeatTransferCoefficient (W/(m^2 K)).",
    ),
];
const CONVECTION_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.ThermalBoundary boundary object.",
)];
const CONVECTION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.ThermalBoundary.convection(id, region, ambientTemperature, coefficient)",
    inputs: &CONVECTION_INPUTS,
    outputs: &CONVECTION_OUTPUTS,
}];
pub const FEA_THERMAL_BOUNDARY_CONVECTION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONVECTION_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const HEAT_FLUX_INPUTS: [BuiltinParamDescriptor; 3] = [
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
        "heatFlux",
        BuiltinParamType::Any,
        "Normal heat flux as fea.HeatFlux (W/m^2).",
    ),
];
const HEAT_FLUX_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "boundary",
    BuiltinParamType::Any,
    "fea.ThermalBoundary boundary object.",
)];
const HEAT_FLUX_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "boundary = fea.ThermalBoundary.heatFlux(id, region, heatFlux)",
    inputs: &HEAT_FLUX_INPUTS,
    outputs: &HEAT_FLUX_OUTPUTS,
}];
pub const FEA_THERMAL_BOUNDARY_HEAT_FLUX_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HEAT_FLUX_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
