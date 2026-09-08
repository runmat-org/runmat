use serde::{Deserialize, Serialize};

use super::quantity::{DynamicViscosity, Frequency, MassDensity, Temperature, TurbulenceIntensity};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermoMechanicalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectroThermalDomain {
    pub reference_temperature: Temperature,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElectromagneticDomain {
    pub reference_frequency: Frequency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CfdDomain {
    pub density: MassDensity,
    pub dynamic_viscosity: DynamicViscosity,
    pub turbulence_intensity: TurbulenceIntensity,
}
