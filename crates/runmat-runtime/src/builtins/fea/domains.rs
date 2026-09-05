use super::*;
use crate::builtins::fea::values::{
    quantity_scalar, DYNAMIC_VISCOSITY_CLASS, FREQUENCY_CLASS, MASS_DENSITY_CLASS,
    TEMPERATURE_CLASS, TURBULENCE_INTENSITY_CLASS,
};

const TYPED_DOMAIN_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
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
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
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

fn domain_object<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    domain: &T,
) -> BuiltinResult<Value> {
    serializable_to_object(builtin, &ERROR_INTERNAL, class_name, domain)
}

#[runtime_builtin(
    name = "fea.ThermoMechanicalDomain.atReferenceTemperature",
    category = "fea",
    summary = "Create a thermo-mechanical domain at a typed reference temperature.",
    keywords = "fea,domain,thermo-mechanical,temperature,typed",
    descriptor(crate::builtins::fea::domains::THERMOMECHANICAL_DOMAIN_DESCRIPTOR),
    integer_audit(TYPED_DOMAIN_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn thermomechanical_domain_builtin(reference_temperature: Value) -> BuiltinResult<Value> {
    domain_object(
        "fea.ThermoMechanicalDomain.atReferenceTemperature",
        FEA_THERMOMECHANICAL_DOMAIN_CLASS,
        &ThermoMechanicalDomain {
            reference_temperature: Temperature::new(quantity_scalar(
                "fea.ThermoMechanicalDomain.atReferenceTemperature",
                &reference_temperature,
                TEMPERATURE_CLASS,
                "kelvin",
            )?),
        },
    )
}

#[runtime_builtin(
    name = "fea.ElectroThermalDomain.atReferenceTemperature",
    category = "fea",
    summary = "Create an electro-thermal domain at a typed reference temperature.",
    keywords = "fea,domain,electro-thermal,temperature,typed",
    descriptor(crate::builtins::fea::domains::ELECTROTHERMAL_DOMAIN_DESCRIPTOR),
    integer_audit(TYPED_DOMAIN_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn electrothermal_domain_builtin(reference_temperature: Value) -> BuiltinResult<Value> {
    domain_object(
        "fea.ElectroThermalDomain.atReferenceTemperature",
        FEA_ELECTROTHERMAL_DOMAIN_CLASS,
        &ElectroThermalDomain {
            reference_temperature: Temperature::new(quantity_scalar(
                "fea.ElectroThermalDomain.atReferenceTemperature",
                &reference_temperature,
                TEMPERATURE_CLASS,
                "kelvin",
            )?),
        },
    )
}

#[runtime_builtin(
    name = "fea.ElectromagneticDomain.atFrequency",
    category = "fea",
    summary = "Create an electromagnetic frequency-domain definition.",
    keywords = "fea,domain,electromagnetic,frequency,typed",
    descriptor(crate::builtins::fea::domains::ELECTROMAGNETIC_DOMAIN_DESCRIPTOR),
    integer_audit(TYPED_DOMAIN_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn electromagnetic_domain_builtin(frequency: Value) -> BuiltinResult<Value> {
    domain_object(
        "fea.ElectromagneticDomain.atFrequency",
        FEA_ELECTROMAGNETIC_DOMAIN_CLASS,
        &ElectromagneticDomain {
            reference_frequency: Frequency::new(quantity_scalar(
                "fea.ElectromagneticDomain.atFrequency",
                &frequency,
                FREQUENCY_CLASS,
                "hertz",
            )?),
        },
    )
}

#[runtime_builtin(
    name = "fea.FluidDomain.newtonian",
    category = "fea",
    summary = "Create a Newtonian fluid domain from typed constitutive properties.",
    keywords = "fea,domain,fluid,cfd,newtonian,density,viscosity,typed",
    descriptor(crate::builtins::fea::domains::FLUID_DOMAIN_DESCRIPTOR),
    integer_audit(TYPED_DOMAIN_INTEGER_AUDIT),
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn fluid_domain_builtin(
    density: Value,
    dynamic_viscosity: Value,
    turbulence_intensity: Value,
) -> BuiltinResult<Value> {
    domain_object(
        "fea.FluidDomain.newtonian",
        FEA_FLUID_DOMAIN_CLASS,
        &CfdDomain {
            density: MassDensity::new(quantity_scalar(
                "fea.FluidDomain.newtonian",
                &density,
                MASS_DENSITY_CLASS,
                "kilograms_per_cubic_meter",
            )?),
            dynamic_viscosity: DynamicViscosity::new(quantity_scalar(
                "fea.FluidDomain.newtonian",
                &dynamic_viscosity,
                DYNAMIC_VISCOSITY_CLASS,
                "pascal_seconds",
            )?),
            turbulence_intensity: TurbulenceIntensity::new(quantity_scalar(
                "fea.FluidDomain.newtonian",
                &turbulence_intensity,
                TURBULENCE_INTENSITY_CLASS,
                "ratio",
            )?),
        },
    )
}
