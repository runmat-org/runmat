//! Signatures, parameters, and integer facts for `fea.FluidDomain` constructors.
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

pub const FEA_FLUID_DOMAIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA domain constructors accept dimensioned quantity objects; integer handling belongs to those quantity constructors.",
};

const NEWTONIAN_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "density",
        BuiltinParamType::Any,
        "Fluid density as positive fea.MassDensity (kg/m^3).",
    ),
    required(
        "dynamicViscosity",
        BuiltinParamType::Any,
        "Dynamic viscosity as positive fea.DynamicViscosity (Pa s).",
    ),
    required(
        "turbulenceIntensity",
        BuiltinParamType::Any,
        "Inlet turbulence intensity as fea.TurbulenceIntensity in [0, 1].",
    ),
];
const NEWTONIAN_OUTPUTS: [BuiltinParamDescriptor; 1] = [required(
    "domain",
    BuiltinParamType::Any,
    "fea.FluidDomain object.",
)];
const NEWTONIAN_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "domain = fea.FluidDomain.newtonian(density, dynamicViscosity, turbulenceIntensity)",
    inputs: &NEWTONIAN_INPUTS,
    outputs: &NEWTONIAN_OUTPUTS,
}];
pub const FEA_FLUID_DOMAIN_NEWTONIAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NEWTONIAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
