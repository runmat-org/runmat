use serde::{Deserialize, Serialize};

fn default_em_source_amplitude_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadKind {
    Force {
        fx: f64,
        fy: f64,
        fz: f64,
    },
    Pressure {
        magnitude_pa: f64,
    },
    BodyForce {
        gx: f64,
        gy: f64,
        gz: f64,
    },
    CurrentDensity {
        jx: f64,
        jy: f64,
        jz: f64,
        #[serde(default)]
        phase_rad: f64,
        #[serde(default = "default_em_source_amplitude_scale")]
        amplitude_scale: f64,
    },
    CoilCurrent {
        current_a: f64,
        #[serde(default)]
        phase_rad: f64,
        #[serde(default = "default_em_source_amplitude_scale")]
        amplitude_scale: f64,
    },
    HeatSource {
        volumetric_w_per_m3: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadCase {
    pub load_id: String,
    pub region_id: String,
    pub kind: LoadKind,
}
