pub mod boundary;
pub mod predicate;
pub mod spatial_index;
pub mod tolerance;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementQuality {
    pub element_id: String,
    pub scaled_jacobian: f64,
    #[serde(default)]
    pub exact_scaled_jacobian: f64,
    pub aspect_ratio: f64,
    pub volume_m3: f64,
}
