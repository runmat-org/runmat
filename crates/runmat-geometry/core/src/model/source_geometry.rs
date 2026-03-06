use serde::{Deserialize, Serialize};

use super::{AssemblyNode, MaterialEvidence};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceGeometryKind {
    Mesh,
    Cad,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceGeometry {
    pub kind: SourceGeometryKind,
    pub assembly: Option<AssemblyNode>,
    pub material_evidence: Vec<MaterialEvidence>,
}
