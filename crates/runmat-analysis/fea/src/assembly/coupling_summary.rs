//! Coupling summaries recorded by structural assembly. The thermomechanical summary names the
//! prescribed temperature profile the assembly applied; solved-state assembly reads
//! the solved thermal field instead and records none.

use runmat_analysis_core::SourceImplementation;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalAssemblySummary {
    /// Number of authored region temperature deltas applied to the thermoelastic load.
    pub region_delta_count: usize,
    /// Sources of the prescribed temperature profile the thermoelastic load used.
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
