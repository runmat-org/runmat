use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialEvidenceConfidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialEvidence {
    pub source_key: String,
    pub normalized_key: String,
    pub value: String,
    pub confidence: MaterialEvidenceConfidence,
    pub unit_basis: Option<String>,
    pub assumptions: Vec<String>,
}
