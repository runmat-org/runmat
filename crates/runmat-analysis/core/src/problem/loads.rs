use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadKind {
    Force { fx: f64, fy: f64, fz: f64 },
    Pressure { magnitude_pa: f64 },
    BodyForce { gx: f64, gy: f64, gz: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadCase {
    pub load_id: String,
    pub region_id: String,
    pub kind: LoadKind,
}
