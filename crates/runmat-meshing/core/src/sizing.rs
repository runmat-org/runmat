use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizingSample {
    pub position_m: [f64; 3],
    pub target_size_m: f64,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshSizingField {
    #[serde(default)]
    pub global_target_size_m: Option<f64>,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub samples: Vec<SizingSample>,
}
