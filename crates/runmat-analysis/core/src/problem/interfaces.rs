use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactInterfaceModel {
    pub penalty_stiffness_scale: f64,
    pub max_penetration_ratio: f64,
    pub friction_coefficient: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisInterfaceKind {
    Contact(ContactInterfaceModel),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisInterface {
    pub interface_id: String,
    pub primary_region_id: String,
    pub secondary_region_id: String,
    pub kind: AnalysisInterfaceKind,
}
