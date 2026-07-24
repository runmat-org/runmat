use serde::{Deserialize, Serialize};

use super::StageEvidence;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizingFieldContract {
    pub field_id: String,
    pub global_target_size_m: f64,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub growth_rate: Option<f64>,
    #[serde(default)]
    pub local_source_count: usize,
    #[serde(default)]
    pub anisotropic_metric_count: usize,
    pub evidence: StageEvidence,
}
