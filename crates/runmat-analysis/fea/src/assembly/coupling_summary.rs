//! Diagnostics for prescribed and reduced coupling branches; solved-state assembly bypasses these.

use runmat_analysis_core::SourceImplementation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalAssemblySummary {
    pub enabled: bool,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub applied_temperature_delta_k: runmat_types::physical::TemperatureDifference,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion,
    pub thermal_strain_scale: f64,
    pub thermal_load_scale: f64,
    pub constitutive_temperature_factor: f64,
    pub constitutive_poisson_coupling: f64,
    pub effective_modulus_scale: f64,
    pub constitutive_material_spread_ratio: f64,
    pub assignment_heterogeneity_index: f64,
    pub spatial_gradient_index: f64,
    pub spatial_coverage_ratio: f64,
    pub region_delta_count: usize,
    pub coupling_fingerprint: u64,
    pub profile_sources: Vec<SourceImplementation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroThermalAssemblySummary {
    pub enabled: bool,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub applied_voltage_v: f64,
    pub base_electrical_conductivity_s_per_m: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub resistive_heating_coefficient: runmat_types::physical::InverseTemperature,
    pub joule_heating_scale: f64,
    pub conductivity_spread_ratio: f64,
    pub region_scale_count: usize,
    pub coupling_fingerprint: u64,
}
