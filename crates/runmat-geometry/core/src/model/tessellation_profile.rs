use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum HealingMode {
    #[default]
    Safe,
    Off,
    Aggressive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TessellationProfile {
    pub profile_id: String,
    pub chord_tolerance: Option<f64>,
    pub angle_tolerance_deg: Option<f64>,
    pub healing_mode: HealingMode,
}

impl Default for TessellationProfile {
    fn default() -> Self {
        Self {
            profile_id: "default-v1".to_string(),
            chord_tolerance: None,
            angle_tolerance_deg: None,
            healing_mode: HealingMode::Safe,
        }
    }
}
