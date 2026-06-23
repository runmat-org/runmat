use serde::{Deserialize, Serialize};

fn default_reference_temperature_k() -> f64 {
    293.15
}

fn default_modulus_temp_coeff_per_k() -> f64 {
    -2.5e-4
}

fn default_thermal_conductivity_w_per_mk() -> f64 {
    45.0
}

fn default_specific_heat_j_per_kgk() -> f64 {
    500.0
}

fn default_thermal_expansion_coefficient_per_k() -> f64 {
    1.2e-5
}

fn default_electrical_conductivity_s_per_m() -> f64 {
    1.0
}

fn default_resistive_heating_coefficient() -> f64 {
    0.0
}

fn default_relative_permittivity() -> f64 {
    1.0
}

fn default_relative_permeability() -> f64 {
    1.0
}

fn default_acoustic_density_kg_per_m3() -> f64 {
    1.225
}
fn default_mechanical_density_kg_per_m3() -> f64 {
    7850.0
}

fn default_speed_of_sound_m_per_s() -> f64 {
    343.0
}

fn default_acoustic_damping_ratio() -> f64 {
    0.02
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConductivityFrequencyPoint {
    pub frequency_hz: f64,
    pub conductivity_scale: f64,
    #[serde(default)]
    pub dispersive_loss_scale: Option<f64>,
    #[serde(default)]
    pub relative_permittivity_scale: Option<f64>,
    #[serde(default)]
    pub relative_permeability_scale: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialMechanicalModel {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    #[serde(default = "default_mechanical_density_kg_per_m3")]
    pub density_kg_per_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialThermalModel {
    #[serde(default = "default_reference_temperature_k")]
    pub reference_temperature_k: f64,
    #[serde(default = "default_modulus_temp_coeff_per_k")]
    pub modulus_temp_coeff_per_k: f64,
    #[serde(default = "default_thermal_conductivity_w_per_mk")]
    pub conductivity_w_per_mk: f64,
    #[serde(default = "default_specific_heat_j_per_kgk")]
    pub specific_heat_j_per_kgk: f64,
    #[serde(default = "default_thermal_expansion_coefficient_per_k")]
    pub expansion_coefficient_per_k: f64,
}

impl Default for MaterialThermalModel {
    fn default() -> Self {
        Self {
            reference_temperature_k: default_reference_temperature_k(),
            modulus_temp_coeff_per_k: default_modulus_temp_coeff_per_k(),
            conductivity_w_per_mk: default_thermal_conductivity_w_per_mk(),
            specific_heat_j_per_kgk: default_specific_heat_j_per_kgk(),
            expansion_coefficient_per_k: default_thermal_expansion_coefficient_per_k(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialAcousticModel {
    #[serde(default = "default_acoustic_density_kg_per_m3")]
    pub density_kg_per_m3: f64,
    #[serde(default = "default_speed_of_sound_m_per_s")]
    pub speed_of_sound_m_per_s: f64,
    #[serde(default = "default_acoustic_damping_ratio")]
    pub damping_ratio: f64,
}

impl Default for MaterialAcousticModel {
    fn default() -> Self {
        Self {
            density_kg_per_m3: default_acoustic_density_kg_per_m3(),
            speed_of_sound_m_per_s: default_speed_of_sound_m_per_s(),
            damping_ratio: default_acoustic_damping_ratio(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialElectricalModel {
    #[serde(default = "default_reference_temperature_k")]
    pub reference_temperature_k: f64,
    #[serde(default = "default_electrical_conductivity_s_per_m")]
    pub conductivity_s_per_m: f64,
    #[serde(default = "default_resistive_heating_coefficient")]
    pub resistive_heating_coefficient: f64,
    #[serde(default = "default_relative_permittivity")]
    pub relative_permittivity: f64,
    #[serde(default = "default_relative_permeability")]
    pub relative_permeability: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conductivity_frequency_response: Vec<ConductivityFrequencyPoint>,
}

impl Default for MaterialElectricalModel {
    fn default() -> Self {
        Self {
            reference_temperature_k: default_reference_temperature_k(),
            conductivity_s_per_m: default_electrical_conductivity_s_per_m(),
            resistive_heating_coefficient: default_resistive_heating_coefficient(),
            relative_permittivity: default_relative_permittivity(),
            relative_permeability: default_relative_permeability(),
            conductivity_frequency_response: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialPlasticModel {
    pub yield_strain: f64,
    pub hardening_modulus_ratio: f64,
    pub saturation_exponent: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    pub mechanical: MaterialMechanicalModel,
    #[serde(default)]
    pub thermal: MaterialThermalModel,
    #[serde(default)]
    pub acoustic: Option<MaterialAcousticModel>,
    #[serde(default)]
    pub electrical: Option<MaterialElectricalModel>,
    #[serde(default)]
    pub plastic: Option<MaterialPlasticModel>,
}
