//! Unchanged descriptor/integer metadata lifted from Runtime domains.rs.
use crate::fea_contract::{descriptors::ERRORS, parameters::OUT_ANY};
use crate::*;

pub const TYPED_DOMAIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed FEA domain constructors accept dimensioned quantity objects; integer handling belongs to those quantity constructors.",
};

const REFERENCE_TEMPERATURE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "reference_temperature",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed fea.Temperature used as the constitutive reference state.",
}];
const FREQUENCY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "frequency",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed positive fea.Frequency for the frequency-domain formulation.",
}];
const FLUID_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "density",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed positive fea.MassDensity.",
    },
    BuiltinParamDescriptor {
        name: "dynamic_viscosity",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed positive fea.DynamicViscosity.",
    },
    BuiltinParamDescriptor {
        name: "turbulence_intensity",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Typed fea.TurbulenceIntensity in [0, 1].",
    },
];

macro_rules! domain_descriptor {
    ($descriptor:ident, $signatures:ident, $label:literal, $inputs:ident) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &$inputs,
            outputs: &OUT_ANY,
        }];
        pub const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

domain_descriptor!(
    THERMOMECHANICAL_DOMAIN_DESCRIPTOR,
    THERMOMECHANICAL_DOMAIN_SIGNATURES,
    "domain = fea.ThermoMechanicalDomain.atReferenceTemperature(referenceTemperature)",
    REFERENCE_TEMPERATURE_INPUTS
);
domain_descriptor!(
    ELECTROTHERMAL_DOMAIN_DESCRIPTOR,
    ELECTROTHERMAL_DOMAIN_SIGNATURES,
    "domain = fea.ElectroThermalDomain.atReferenceTemperature(referenceTemperature)",
    REFERENCE_TEMPERATURE_INPUTS
);
domain_descriptor!(
    ELECTROMAGNETIC_DOMAIN_DESCRIPTOR,
    ELECTROMAGNETIC_DOMAIN_SIGNATURES,
    "domain = fea.ElectromagneticDomain.atFrequency(frequency)",
    FREQUENCY_INPUTS
);
domain_descriptor!(
    FLUID_DOMAIN_DESCRIPTOR,
    FLUID_DOMAIN_SIGNATURES,
    "domain = fea.FluidDomain.newtonian(density, dynamicViscosity, turbulenceIntensity)",
    FLUID_INPUTS
);
