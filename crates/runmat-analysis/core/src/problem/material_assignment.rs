use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceConfidence {
    Verified,
    Probable,
    Inferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialAssignment {
    pub region_id: String,
    pub expected_material_id: String,
    pub assigned_material_id: String,
    pub confidence: EvidenceConfidence,
}
