use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceEntityKind {
    Body,
    Face,
    Edge,
    Region,
    Mesh,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshEntityProvenance {
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    pub source_entity_kind: SourceEntityKind,
    pub source_entity_id: String,
    #[serde(default)]
    pub region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisMeshProvenance {
    pub algorithm: String,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
    #[serde(default)]
    pub source_geometry_sha256: Option<String>,
}
