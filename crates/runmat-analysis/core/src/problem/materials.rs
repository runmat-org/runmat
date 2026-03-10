use serde::{Deserialize, Serialize};

fn default_reference_temperature_k() -> f64 {
    293.15
}

fn default_modulus_temp_coeff_per_k() -> f64 {
    -2.5e-4
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    #[serde(default = "default_reference_temperature_k")]
    pub reference_temperature_k: f64,
    #[serde(default = "default_modulus_temp_coeff_per_k")]
    pub modulus_temp_coeff_per_k: f64,
}
