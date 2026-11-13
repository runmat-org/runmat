use super::*;
use crate::builtins::fea::values::{
    quantity_scalar, DYNAMIC_VISCOSITY_CLASS, FREQUENCY_CLASS, MASS_DENSITY_CLASS,
    TEMPERATURE_CLASS, TURBULENCE_INTENSITY_CLASS,
};
use runmat_analysis_core::ThermalDomain;

fn domain_object<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    domain: &T,
) -> BuiltinResult<Value> {
    serializable_to_object(builtin, &ERROR_INTERNAL, class_name, domain)
}

#[runtime_builtin(
    name = "fea.ThermalDomain.atReferenceTemperature",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn thermal_domain_builtin(reference_temperature: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_THERMAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR,
        "fea.ThermalDomain.atReferenceTemperature",
    )?;
    domain_object(
        "fea.ThermalDomain.atReferenceTemperature",
        FEA_THERMAL_DOMAIN_CLASS,
        &ThermalDomain {
            reference_temperature: Temperature::new(quantity_scalar(
                "fea.ThermalDomain.atReferenceTemperature",
                &reference_temperature,
                TEMPERATURE_CLASS,
                "kelvin",
            )?),
        },
    )
}

#[runtime_builtin(
    name = "fea.ThermoMechanicalDomain.atReferenceTemperature",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn thermomechanical_domain_builtin(reference_temperature: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_THERMO_MECHANICAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR,
        "fea.ThermoMechanicalDomain.atReferenceTemperature",
    )?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn electrothermal_domain_builtin(reference_temperature: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ELECTRO_THERMAL_DOMAIN_AT_REFERENCE_TEMPERATURE_DESCRIPTOR,
        "fea.ElectroThermalDomain.atReferenceTemperature",
    )?;
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn electromagnetic_domain_builtin(frequency: Value) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_ELECTROMAGNETIC_DOMAIN_AT_FREQUENCY_DESCRIPTOR,
        "fea.ElectromagneticDomain.atFrequency",
    )?;
    domain_object(
        "fea.ElectromagneticDomain.atFrequency",
        FEA_ELECTROMAGNETIC_DOMAIN_CLASS,
        &ElectromagneticDomain::Harmonic {
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
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::domains"
)]
pub async fn fluid_domain_builtin(
    density: Value,
    dynamic_viscosity: Value,
    turbulence_intensity: Value,
) -> BuiltinResult<Value> {
    crate::builtins::fea::constraints::fixed_outputs(
        &runmat_builtins::FEA_FLUID_DOMAIN_NEWTONIAN_DESCRIPTOR,
        "fea.FluidDomain.newtonian",
    )?;
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
            rheology: None,
            porous_zones: Vec::new(),
        },
    )
}
