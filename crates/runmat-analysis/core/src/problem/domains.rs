use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoRegionTemperatureDelta {
    pub region_id: String,
    pub temperature_delta_k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoTimeProfilePoint {
    pub normalized_time: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermoFieldInterpolationMode {
    Linear,
    Step,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoFieldSource {
    pub source_id: String,
    pub revision: u32,
    #[serde(default)]
    pub interpolation_mode: Option<ThermoFieldInterpolationMode>,
    #[serde(default)]
    pub expected_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermoMechanicalDomain {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_temperature_delta_k: f64,
    #[serde(default)]
    pub field_artifact_id: Option<String>,
    #[serde(default)]
    pub field_source: Option<ThermoFieldSource>,
    #[serde(default)]
    pub region_temperature_deltas: Vec<ThermoRegionTemperatureDelta>,
    #[serde(default)]
    pub time_profile: Vec<ThermoTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroRegionConductivityScale {
    pub region_id: String,
    pub conductivity_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroTimeProfilePoint {
    pub normalized_time: f64,
    pub current_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectroThermalDomain {
    pub enabled: bool,
    pub reference_temperature_k: f64,
    pub applied_voltage_v: f64,
    #[serde(default)]
    pub region_conductivity_scales: Vec<ElectroRegionConductivityScale>,
    #[serde(default)]
    pub time_profile: Vec<ElectroTimeProfilePoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElectromagneticDomain {
    pub enabled: bool,
    pub reference_frequency_hz: f64,
    pub applied_current_a: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CfdSolveFamily {
    SteadyState,
    Transient,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CfdTimeProfilePoint {
    pub normalized_time: f64,
    pub inlet_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CfdDomain {
    pub enabled: bool,
    pub solve_family: CfdSolveFamily,
    pub reference_density_kg_per_m3: f64,
    pub dynamic_viscosity_pa_s: f64,
    pub inlet_velocity_m_per_s: f64,
    pub turbulence_intensity: f64,
    #[serde(default)]
    pub time_profile: Vec<CfdTimeProfilePoint>,
}
