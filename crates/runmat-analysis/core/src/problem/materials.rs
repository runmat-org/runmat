use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialModel {
    pub material_id: String,
    pub name: String,
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
}
